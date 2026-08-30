import sys
from pathlib import Path
from pprint import pprint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests

from src.agent.llm import load_project_environment
from src.agent.llm import OpenAICompatibleChatProvider


def direct_probe() -> None:
    import os

    load_project_environment()
    api_key = os.environ.get("ZAI_API_KEY")
    if not api_key:
        raise RuntimeError("ZAI_API_KEY is not set")
    response = requests.post(
        "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        json={
            "model": "glm-5.3",
            "messages": [
                {"role": "system", "content": "Return concise JSON only."},
                {
                    "role": "user",
                    "content": (
                        "Return exactly this JSON object with different short strings: "
                        '{"objective":"x","questions":["x"],"feature_hypotheses":["x"],'
                        '"required_inputs":["x"],"leakage_risks":["x"],'
                        '"expected_artifacts":["x"]}'
                    ),
                },
            ],
            "stream": False,
            "temperature": 1,
            "thinking": {"type": "enabled"},
            "reasoning_effort": "low",
        },
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=60,
    )
    print("direct_status", response.status_code)
    payload = response.json()
    pprint(payload)


def main() -> None:
    direct_probe()
    provider = OpenAICompatibleChatProvider(
        {
            "provider": "glm",
            "model": "glm-5.3",
            "base_url": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
            "api_key_env": "ZAI_API_KEY",
            "temperature": 1,
            "thinking": "low",
            "max_output_tokens": 1600,
            "max_retries": 1,
        }
    )
    result = provider.complete(
        role="eda_researcher",
        instructions="Return structured JSON only.",
        prompt=(
            "Create a tiny EDA plan for KuaiRand-Pure within-user long_view ranking. "
            "Keep it concise."
        ),
        schema_name="eda_research_plan",
        max_output_tokens=1600,
        max_retries=1,
    )
    print(
        {
            "response_id_present": bool(result.response_id),
            "model": result.model,
            "keys": sorted(result.data.keys()),
            "total_tokens": result.usage.total_tokens,
        }
    )


if __name__ == "__main__":
    main()
