"""LLM-based threat summarization via AWS Bedrock (Claude)."""

import os

import pandas as pd
from langchain_aws import BedrockLLM
from langchain_core.prompts import PromptTemplate

_PROMPT_TEMPLATE = (
    "You are a maritime and air traffic intelligence analyst.\n"
    "Given the following anomalous vessel or aircraft data flagged by our detection system,\n"
    "generate a concise intelligence summary (3-5 sentences) suitable for an analyst brief.\n"
    "Include: 1) anomaly type detected, 2) approximate location, 3) threat level LOW/MEDIUM/HIGH"
    " with one sentence justification, 4) recommended follow-up action.\n\n"
    "Anomaly data:\n"
    "{anomaly_data}\n\n"
    "Intelligence Summary:"
)

_ID_COLS = ["mmsi", "icao24"]


def _get_id_col(df: pd.DataFrame) -> str | None:
    """Return the first available identifier column name, or None."""
    for col in _ID_COLS:
        if col in df.columns:
            return col
    return None


def _flag_cols(df: pd.DataFrame) -> list[str]:
    """Return all columns whose name ends with '_flag'."""
    return [c for c in df.columns if c.endswith("_flag")]


class ThreatSummarizer:
    """Generates intelligence summaries for anomalous events via AWS Bedrock (Claude).

    Attributes:
        model_id: Bedrock model identifier.
        region_name: AWS region for Bedrock API calls.
    """

    def __init__(self) -> None:
        """Initialise Bedrock LLM and LangChain prompt chain from environment variables."""
        self.model_id: str = os.getenv(
            "BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0"
        )
        self.region_name: str = os.getenv("AWS_REGION", "us-east-1")

        llm = BedrockLLM(
            model_id=self.model_id,
            region_name=self.region_name,
            model_kwargs={"max_tokens": 512, "temperature": 0.3},
        )
        prompt = PromptTemplate(input_variables=["anomaly_data"], template=_PROMPT_TEMPLATE)
        self._chain = prompt | llm

    def summarize(self, anomalous_df: pd.DataFrame) -> str:
        """Generate an intelligence summary for the highest-scoring anomalies.

        Args:
            anomalous_df: DataFrame containing anomaly detections with a ``threat_score`` column.

        Returns:
            Intelligence summary string, or an error message if the LLM call fails.
        """
        try:
            top = anomalous_df.nlargest(5, "threat_score")
            id_col = _get_id_col(top)
            keep = (
                ([id_col] if id_col else [])
                + ["latitude", "longitude", "threat_score"]
                + _flag_cols(top)
            )
            keep = [c for c in keep if c in top.columns]
            snippet = top[keep].to_string(index=False)
            result = self._chain.invoke({"anomaly_data": snippet})
            return str(result)
        except Exception as exc:  # noqa: BLE001
            return f"LLM summarization unavailable: {exc}"

    def summarize_batch(
        self, anomalous_df: pd.DataFrame, max_items: int = 10
    ) -> list[dict]:
        """Generate individual summaries for the top ``max_items`` anomalous rows.

        Args:
            anomalous_df: DataFrame containing anomaly detections with a ``threat_score`` column.
            max_items: Maximum number of rows to summarize.

        Returns:
            List of dicts with keys ``id``, ``threat_score``, and ``summary``.
        """
        top = anomalous_df.nlargest(max_items, "threat_score")
        id_col = _get_id_col(top)
        results: list[dict] = []
        for _, row in top.iterrows():
            row_id = str(row[id_col]) if id_col and id_col in row.index else "unknown"
            single = row.to_frame().T
            summary = self.summarize(single)
            results.append(
                {
                    "id": row_id,
                    "threat_score": float(row["threat_score"]),
                    "summary": summary,
                }
            )
        return results


class MockThreatSummarizer:
    """Drop-in replacement for ThreatSummarizer when AWS credentials are unavailable.

    Returns realistic hardcoded summaries without making any external API calls.
    """

    _MOCK_SUMMARY = (
        "ANOMALY DETECTED — VESSEL AIS SPOOFING / DARK VESSEL ACTIVITY: "
        "An unidentified vessel (MMSI 123456789) was flagged for AIS signal loss "
        "lasting approximately 4 hours while transiting the Strait of Hormuz "
        "(lat 26.31°N, lon 56.42°E). "
        "Threat Level: HIGH — the combination of loitering behaviour, "
        "abnormal speed deviations, and proximity to a known maritime chokepoint "
        "indicates a high probability of deliberate signal suppression consistent "
        "with sanctions evasion or pre-positioning for hostile activity. "
        "Recommended action: Task a maritime patrol asset or coordinate with regional "
        "coast-guard authorities for visual identification; flag vessel in watch-list "
        "and cross-reference against recent port-state control records."
    )

    def summarize(self, anomalous_df: pd.DataFrame) -> str:  # noqa: ARG002
        """Return a hardcoded mock intelligence summary.

        Args:
            anomalous_df: Ignored; present for interface compatibility.

        Returns:
            A realistic mock intelligence summary string.
        """
        return self._MOCK_SUMMARY

    def summarize_batch(
        self, anomalous_df: pd.DataFrame, max_items: int = 10
    ) -> list[dict]:
        """Return mock individual summaries for up to ``max_items`` rows.

        Args:
            anomalous_df: DataFrame containing anomaly detections with a ``threat_score`` column.
            max_items: Maximum number of rows to summarize.

        Returns:
            List of dicts with keys ``id``, ``threat_score``, and ``summary``.
        """
        top = anomalous_df.nlargest(max_items, "threat_score")
        id_col = _get_id_col(top)
        results: list[dict] = []
        for _, row in top.iterrows():
            row_id = str(row[id_col]) if id_col and id_col in row.index else "unknown"
            results.append(
                {
                    "id": row_id,
                    "threat_score": float(row["threat_score"]),
                    "summary": self._MOCK_SUMMARY,
                }
            )
        return results


def get_summarizer() -> ThreatSummarizer | MockThreatSummarizer:
    """Return a ThreatSummarizer if AWS credentials are present, else MockThreatSummarizer.

    Returns:
        An instance of ThreatSummarizer or MockThreatSummarizer.
    """
    if os.getenv("AWS_ACCESS_KEY_ID"):
        return ThreatSummarizer()
    return MockThreatSummarizer()


# Module-level default summarizer instance
summarizer: ThreatSummarizer | MockThreatSummarizer = get_summarizer()
