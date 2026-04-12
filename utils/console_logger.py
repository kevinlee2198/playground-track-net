"""Console logger for real-time match output."""

from datetime import datetime
from typing import Dict, List, Optional


class ConsoleMatchLogger:
    """Pretty console output for match progress."""

    def __init__(self, player_names: List[str], sport: str = "tennis"):
        """
        Initialize match logger.

        Args:
            player_names: List of player names (e.g., ["Player 1", "Player 2"])
            sport: Sport type ('tennis', 'badminton', 'pickleball')
        """
        self.player_names = player_names
        self.sport = sport
        self.start_time = datetime.now()
        self._print_header()

    def _print_header(self):
        """Print match header."""
        print("\n" + "═" * 70)
        print("              TrackNet Match Analysis".center(70))
        print(f"  {self.sport.title()} - {datetime.now().strftime('%Y-%m-%d')}".center(70))
        print(
            f"  {self.player_names[0]} vs {self.player_names[1]}".center(70)
        )
        print("═" * 70 + "\n")

    def log_event(self, event: Dict, frame_id: int, timestamp: float):
        """
        Log single event to console.

        Args:
            event: Event dictionary with 'type' and event-specific data
            frame_id: Frame number
            timestamp: Time in seconds from start
        """
        time_str = f"{timestamp:06.1f}s"
        frame_str = f"Frame {frame_id:04d}"

        if event["type"] == "serve_start":
            player_name = self.player_names[event.get("player_id", 0)]
            print(f"\n{'─' * 70}")
            print(f"  SERVE - {player_name}")
            print("─" * 70)

        elif event["type"] == "bounce":
            data = event.get("data", {})
            zone = data.get("zone", "unknown")
            valid = "✓" if data.get("valid", True) else "✗"
            print(f"{frame_str} | {time_str} | [BOUNCE] {zone} {valid}")

        elif event["type"] == "hit":
            player_id = event.get("player_id", 0)
            player_name = self.player_names[player_id]
            data = event.get("data", {})
            shot_type = data.get("shot_type", "unknown")
            print(f"{frame_str} | {time_str} | [HIT] {player_name} - {shot_type}")

        elif event["type"] == "rally_end":
            winner_id = event.get("winner", 0)
            winner_name = self.player_names[winner_id]
            reason = event.get("reason", "unknown")
            print(f"\n  🎾 Point: {winner_name} ({reason})\n")

    def log_score(self, score_string: str):
        """
        Log current score.

        Args:
            score_string: Formatted score string (e.g., "30-15 | Games: 0-0")
        """
        print(f"  Score: {score_string}\n")

    def log_stats(self, player_id: int, stats: Dict):
        """
        Log player statistics.

        Args:
            player_id: Player index (0 or 1)
            stats: Statistics dictionary
        """
        player_name = self.player_names[player_id]
        print(f"  {player_name} Stats:")

        # Core stats
        print(f"    Points Won: {stats.get('points_won', 0)}")
        print(f"    Aces: {stats.get('aces', 0)}")
        print(f"    Winners: {stats.get('winners', 0)}")
        print(f"    Errors: {stats.get('unforced_errors', 0)}")

        # Serve stats
        first_serves_in = stats.get("first_serves_in", 0)
        first_serves_total = stats.get("first_serves_total", 0)
        if first_serves_total > 0:
            first_pct = first_serves_in / first_serves_total * 100
            print(
                f"    First Serve: {first_serves_in}/{first_serves_total} ({first_pct:.0f}%)"
            )
        else:
            print("    First Serve: 0/0 (—)")

        print()

    def log_final_summary(
        self,
        final_score: str,
        winner_id: int,
        player_stats: List[Dict],
        duration_seconds: int,
    ):
        """
        Print final match summary.

        Args:
            final_score: Final score string (e.g., "6-4, 6-3")
            winner_id: Winner player ID
            player_stats: List of stats dicts for each player
            duration_seconds: Match duration in seconds
        """
        print("\n" + "═" * 70)
        print("                    MATCH COMPLETE".center(70))
        print("═" * 70 + "\n")

        winner_name = self.player_names[winner_id]
        print(f"  Final Score: {final_score}")
        print(f"  Winner: {winner_name}")

        # Format duration
        hours = duration_seconds // 3600
        minutes = (duration_seconds % 3600) // 60
        if hours > 0:
            print(f"  Duration: {hours}h {minutes}m\n")
        else:
            print(f"  Duration: {minutes}m\n")

        # Print detailed stats for each player
        for player_id, stats in enumerate(player_stats):
            self._print_player_final_stats(player_id, stats)

    def _print_player_final_stats(self, player_id: int, stats: Dict):
        """
        Print detailed final stats for one player.

        Args:
            player_id: Player index
            stats: Statistics dictionary
        """
        player_name = self.player_names[player_id]
        print(f"  {player_name} Final Stats:")
        print(f"    Points Won: {stats.get('points_won', 0)}")
        print(f"    Aces: {stats.get('aces', 0)}")
        print(f"    Double Faults: {stats.get('double_faults', 0)}")
        print(f"    Winners: {stats.get('winners', 0)}")
        print(f"    Unforced Errors: {stats.get('unforced_errors', 0)}")

        # First serve percentage
        first_serves_in = stats.get("first_serves_in", 0)
        first_serves_total = stats.get("first_serves_total", 0)
        if first_serves_total > 0:
            first_pct = first_serves_in / first_serves_total * 100
            print(
                f"    First Serve: {first_pct:.0f}% ({first_serves_in}/{first_serves_total})"
            )

        # Ace percentage
        aces = stats.get("aces", 0)
        service_points_total = stats.get("service_points_total", 0)
        if service_points_total > 0:
            ace_pct = aces / service_points_total * 100
            print(f"    Ace %: {ace_pct:.1f}%")

        # Winner rate
        winners = stats.get("winners", 0)
        total_shots = stats.get("total_shots", 0)
        if total_shots > 0:
            winner_rate = winners / total_shots * 100
            print(f"    Winner Rate: {winner_rate:.1f}%")

        print()
