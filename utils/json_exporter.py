"""JSON exporter for match data."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class MatchJSONExporter:
    """Export match data to JSON files."""

    @staticmethod
    def export_full(
        output_path: str,
        match_metadata: Dict,
        players: List[Dict],
        final_score: Dict,
        statistics: Dict,
        events: List[Dict],
        rally_log: List[Dict],
        processing_info: Dict,
    ):
        """
        Export complete match data to JSON.

        Args:
            output_path: Path to output JSON file
            match_metadata: Match metadata dict
            players: List of player info dicts
            final_score: Final score dict
            statistics: Player statistics dict
            events: List of event dicts
            rally_log: List of rally dicts
            processing_info: Processing info dict
        """
        data = {
            "match_metadata": match_metadata,
            "players": players,
            "final_score": final_score,
            "statistics": statistics,
            "events": events,
            "rally_log": rally_log,
            "processing_info": processing_info,
        }

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Write JSON
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\nMatch data saved to: {output_path}")

    @staticmethod
    def export_minimal(
        output_path: str,
        match_id: str,
        date: str,
        sport: str,
        player_names: List[str],
        winner_id: int,
        final_score: str,
        player_stats: List[Dict],
    ):
        """
        Export minimal match data (lighter format for quick integration).

        Args:
            output_path: Path to output JSON file
            match_id: Unique match identifier
            date: Match date (ISO format)
            sport: Sport type
            player_names: List of player names
            winner_id: Winner player ID (0 or 1)
            final_score: Final score string
            player_stats: List of player stats dicts
        """
        data = {
            "match_id": match_id,
            "date": date,
            "sport": sport,
            "players": player_names,
            "winner": winner_id,
            "final_score": final_score,
            "player_stats": player_stats,
        }

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Write JSON
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Minimal match data saved to: {output_path}")
