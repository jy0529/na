# ReAct Pattern

from llm import OpenAIClient, LLMSettings
from typing import Dict, Any, List, Optional, AsyncIterator
import asyncio
from datetime import datetime
from dataclasses import dataclass


@dataclass
class ReActConfig:
    """ReAct 配置"""
    thinking_model: str = "deepseek/deepseek-v3.2"
    action_model: str = "xiaomi/mimo-v2-flash:free"
    continue_model: str = "deepseek/deepseek-v3.2"
    synthesis_model: str = "xiaomi/mimo-v2-flash:free"
    temperature: float = 0.3
    min_iterations: int = 1
    max_iterations: int = 5


async def extract_stream_content(
    stream: AsyncIterator[Any],
    print_output: bool = True
) -> str:
    """
    从流式响应中提取内容
    
    Args:
        stream: 异步生成器流
        print_output: 是否打印输出
        
    Returns:
        提取的文本内容
    """
    content = ""
    try:
        async for chunk in stream:
            if hasattr(chunk, 'choices') and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    text = delta.content
                    if print_output:
                        print(text, end='', flush=True)
                    content += text
    except Exception as e:
        if print_output:
            print(f"\n❌ Error extracting stream content: {e}")
        raise
    return content


async def think_phase(
    llm_client: OpenAIClient,
    user_query: str,
    current_datetime: str,
    config: ReActConfig
) -> str:
    """
    思考阶段：分析用户查询并制定计划
    
    Args:
        llm_client: LLM 客户端
        user_query: 用户查询
        current_datetime: 当前 UTC 日期时间
        config: ReAct 配置
        
    Returns:
        思考内容
    """
    print("💭 [THOUGHT PHASE] Starting...")
    print("-" * 60)
    
    thinking_system_prompt = f"""
        You are a helpful assistant that can think and reason about the user's query. 
        You will be given a user query and you will need to think about the user's query and reason about the user's query.
        You will then return your thoughts.
        
        Current date and time: {current_datetime}

        IMPORTANT: This is a REASONING step. You have NO tools available.
        - Output ONLY your reasoning and decision (search/no_search).
        - Do NOT output any tool calls, XML tags, JSON, or function call stubs.
        - Do NOT use <function_calls>, <invoke>, <web_fetch>, or similar markup.
        - Simply provide your reasoning in plain text.
        """
    
    try:
        thought_stream = llm_client.stream_chat(
            model=config.thinking_model,
            temperature=config.temperature,
            messages=[
                {
                    "role": "system",
                    "content": thinking_system_prompt
                },
                {
                    "role": "user",
                    "content": user_query
                }
            ],
        )
        thought_content = await extract_stream_content(thought_stream)
        print(f"\n{'='*60}")
        print("✅ [THOUGHT PHASE] Completed")
        print(f"{'='*60}\n")
        return thought_content
    except Exception as e:
        print(f"\n❌ [THOUGHT PHASE] Error: {e}")
        print(f"{'='*60}\n")
        raise


async def act_phase(
    llm_client: OpenAIClient,
    user_query: str,
    observations: List[str],
    current_thought: str,
    current_datetime: str,
    config: ReActConfig
) -> str:
    """
    行动阶段：执行计划并获取观察结果
    
    Args:
        llm_client: LLM 客户端
        user_query: 用户查询
        observations: 之前的观察列表
        current_thought: 当前思考内容
        current_datetime: 当前 UTC 日期时间
        config: ReAct 配置
        
    Returns:
        观察内容
    """
    print("⚡ [ACTION PHASE] Starting...")
    print("-" * 60)
    
    observations_text = '\n'.join([f"- {o}" for o in observations]) if observations else "None"
    
    action_system_prompt = f"""
        ACT on this plan: {current_thought}
        
        Constraints:
        - Execute the next step with available tools.
        - Keep response in the SAME language as the user's query.
        - Keep actions atomic. If you used a tool that fetched information, add a brief 1–2 sentence summary of the key finding.
        
        Current date and time: {current_datetime}
        Current thought: {current_thought}
        Previous observations: {observations_text}
        """
    
    try:
        action_stream = llm_client.stream_chat(
            model=config.action_model,
            temperature=config.temperature,
            messages=[
                {
                    "role": "system",
                    "content": action_system_prompt
                },
                {
                    "role": "user",
                    "content": f"User query: {user_query}"
                }
            ]
        )
        observation_content = await extract_stream_content(action_stream)
        print(f"\n{'='*60}")
        print("✅ [ACTION PHASE] Completed")
        print(f"{'='*60}\n")
        return observation_content
    except Exception as e:
        print(f"\n❌ [ACTION PHASE] Error: {e}")
        print(f"{'='*60}\n")
        raise


async def decide_phase(
    llm_client: OpenAIClient,
    user_query: str,
    thoughts: List[str],
    observations: List[str],
    current_datetime: str,
    config: ReActConfig,
    current_iteration: int
) -> bool:
    """
    决策阶段：决定是否继续迭代
    
    Args:
        llm_client: LLM 客户端
        user_query: 用户查询
        thoughts: 思考列表
        observations: 观察列表
        current_datetime: 当前 UTC 日期时间
        config: ReAct 配置
        current_iteration: 当前迭代次数
        
    Returns:
        是否继续（True/False）
    """
    print("🤔 [DECISION PHASE] Checking if we should continue...")
    
    # 如果还没达到最小迭代次数，直接返回 True
    if current_iteration < config.min_iterations:
        print(f"✅ Decision: Continue (Minimum iterations not reached: {current_iteration}/{config.min_iterations})")
        return True
    
    continue_prompt = f"""
        You are a helpful assistant that can decide if we should continue.
        
        Current date and time: {current_datetime}
        Current iteration: {current_iteration}/{config.max_iterations}
        User query: {{user_query}}
        recent_thoughts: {{thoughts}}
        recent_observations: {{observations}}
        If we should continue, return "yes". If we should not continue, return "no".
        Do not include any other text in your response.
        """
    
    try:
        continue_stream = llm_client.stream_chat(
            model=config.continue_model,
            temperature=config.temperature,
            messages=[
                {
                    "role": "system",
                    "content": continue_prompt.format(
                        user_query=user_query,
                        thoughts='\n'.join(thoughts),
                        observations='\n'.join(observations)
                    )
                }
            ]
        )
        continue_response = await extract_stream_content(continue_stream, print_output=True)
        
        should_continue = continue_response and "yes" in continue_response.lower()
        if should_continue:
            print(f"✅ Decision: Continue (Response: {continue_response.strip()})")
        else:
            print(f"🛑 Decision: Stop (Response: {continue_response.strip()})")
        return should_continue
    except Exception as e:
        print(f"❌ [DECISION PHASE] Error: {e}, defaulting to stop")
        return False


async def synthesize_phase(
    llm_client: OpenAIClient,
    user_query: str,
    thoughts: List[str],
    observations: List[str],
    current_datetime: str,
    config: ReActConfig
) -> str:
    """
    合成阶段：生成最终响应
    
    Args:
        llm_client: LLM 客户端
        user_query: 用户查询
        thoughts: 思考列表
        observations: 观察列表
        current_datetime: 当前 UTC 日期时间
        config: ReAct 配置
        
    Returns:
        最终响应
    """
    print(f"\n{'='*60}")
    print("📝 [SYNTHESIS PHASE] Generating final response...")
    print("-" * 60)
    
    synthesis_system_prompt = f"""
    You are a helpful assistant that can synthesize the thoughts and observations into a final response.
    You will be given the thoughts and observations.
    You will then return the final response.
    
    Current date and time: {current_datetime}
    User query: {{user_query}}
    Recent thoughts: {{thoughts}}
    Recent observations: {{observations}}
    """
    
    try:
        synthesis_stream = llm_client.stream_chat(
            model=config.synthesis_model,
            temperature=config.temperature,
            messages=[
                {
                    "role": "system",
                    "content": synthesis_system_prompt.format(
                        user_query=user_query,
                        thoughts='\n'.join(thoughts),
                        observations='\n'.join(observations)
                    )
                }
            ],
        )
        final_response = await extract_stream_content(synthesis_stream)
        print(f"\n{'='*60}")
        print("✅ [SYNTHESIS PHASE] Completed")
        print(f"{'='*60}")
        return final_response
    except Exception as e:
        print(f"\n❌ [SYNTHESIS PHASE] Error: {e}")
        print(f"{'='*60}")
        raise


async def react_loop(
    user_query: str,
    config: Optional[ReActConfig] = None,
    llm_client: Optional[OpenAIClient] = None,
    min_iterations: Optional[int] = None,
    max_iterations: Optional[int] = None,
) -> Dict[str, Any]:
    """
    ReAct Pattern 主循环
    
    Args:
        user_query: 用户查询
        config: ReAct 配置（如果为 None 则使用默认配置）
        llm_client: LLM 客户端（如果为 None 则创建新客户端）
        
    Returns:
        包含最终响应、思考、观察和迭代次数的字典
    """
    # 使用默认配置或合并参数
    if config is None:
        config = ReActConfig()
        if min_iterations is not None:
            config.min_iterations = min_iterations
        if max_iterations is not None:
            config.max_iterations = max_iterations
    
    # 创建 LLM 客户端（如果未提供）
    if llm_client is None:
        llm_settings = LLMSettings()
        llm_client = OpenAIClient(llm_settings=llm_settings)
    
    thoughts: List[str] = []
    observations: List[str] = []
    iterations = 0
    
    # 主循环
    while iterations < config.max_iterations:
        # 每次迭代更新日期时间
        current_datetime = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        print(f"\n{'='*60}")
        print(f"🔄 Iteration {iterations + 1}/{config.max_iterations}")
        print(f"{'='*60}\n")
        
        try:
            # 思考阶段
            thought_content = await think_phase(
                llm_client, user_query, current_datetime, config
            )
            thoughts.append(thought_content)
            
            # 行动阶段
            observation_content = await act_phase(
                llm_client, user_query, thoughts, observations, current_datetime, config
            )
            observations.append(observation_content)
            
            # 决策阶段
            should_continue = await decide_phase(
                llm_client, user_query, thoughts, observations, 
                current_datetime, config, iterations + 1
            )
            
            iterations += 1
            
            if not should_continue:
                break
                
        except Exception as e:
            print(f"\n❌ Error in iteration {iterations + 1}: {e}")
            # 如果是最小迭代次数内的错误，可以选择继续或停止
            if iterations < config.min_iterations:
                print("⚠️  Warning: Error occurred before minimum iterations reached")
                observations.append(str(e))
            raise
    
    # 合成最终响应
    current_datetime = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    final_response = await synthesize_phase(
        llm_client, user_query, thoughts, observations, current_datetime, config
    )
    
    print(f"\n🎉 Final Response:\n{final_response}\n")
    
    return {
        "final_response": final_response,
        "thoughts": thoughts,
        "observations": observations,
        "iterations": iterations,
    }


if __name__ == "__main__":
    asyncio.run(react_loop(user_query="最近有什么电影好看？"))
