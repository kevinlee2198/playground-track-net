"""Scoring engines for tennis, badminton, and pickleball."""

from typing import Optional


class TennisScorer:
    """Tennis scoring engine with proper point/game/set/match logic.

    Implements standard tennis rules:
    - Points: 0, 15, 30, 40, game (with deuce/advantage)
    - Games: First to 6 with 2-game lead (or tiebreak at 6-6)
    - Sets: Best of 3 or best of 5
    - Tiebreak: First to 7 points with 2-point lead
    """

    def __init__(
        self,
        player1_name: str = "Player 1",
        player2_name: str = "Player 2",
        best_of: int = 3,
    ):
        """Initialize tennis scorer.

        Args:
            player1_name: Name of player 1
            player2_name: Name of player 2
            best_of: Best of 3 or 5 sets (default: 3)
        """
        self.player_names = [player1_name, player2_name]
        self.best_of = best_of
        self.reset_match()

    def reset_match(self):
        """Reset match to initial state."""
        self.score = {
            0: {"sets": 0, "games": 0, "points": 0},
            1: {"sets": 0, "games": 0, "points": 0},
        }
        self.server = 0  # 0 = player 0 serves, 1 = player 1 serves
        self.tiebreak = False
        self.match_over = False
        self.rally_history = []
        self.stats = {
            0: self._init_player_stats(),
            1: self._init_player_stats(),
        }

    def _init_player_stats(self) -> dict:
        """Initialize player statistics."""
        return {
            "points_won": 0,
            "aces": 0,
            "double_faults": 0,
            "winners": 0,
            "unforced_errors": 0,
            "first_serves_in": 0,
            "first_serves_total": 0,
            "service_points_total": 0,
            "total_shots": 0,
        }

    def award_point(self, winner_id: int, reason: str = ""):
        """Award point to winner.

        Args:
            winner_id: Player who won the point (0 or 1)
            reason: Reason for point (ace, winner, error, out, etc.)
        """
        if self.match_over:
            return

        loser_id = 1 - winner_id

        # Update stats
        self.stats[winner_id]["points_won"] += 1
        if reason == "ace":
            self.stats[winner_id]["aces"] += 1
        elif reason == "winner":
            self.stats[winner_id]["winners"] += 1
        elif reason in ["out", "fault", "net"]:
            self.stats[loser_id]["unforced_errors"] += 1
        elif reason == "double_fault":
            self.stats[loser_id]["double_faults"] += 1

        # Tiebreak scoring
        if self.tiebreak:
            self.score[winner_id]["points"] += 1

            # Switch server every 2 points in tiebreak
            total_points = sum(self.score[p]["points"] for p in [0, 1])
            if total_points % 2 == 1:
                self.server = 1 - self.server

            # Tiebreak win: first to 7 with 2-point lead
            if (
                self.score[winner_id]["points"] >= 7
                and self.score[winner_id]["points"] - self.score[loser_id]["points"]
                >= 2
            ):
                self._award_game(winner_id)
                self.tiebreak = False
                self.score[0]["points"] = 0
                self.score[1]["points"] = 0
        else:
            # Regular point scoring
            self.score[winner_id]["points"] += 1

            # Check for game win
            p_win = self.score[winner_id]["points"]
            p_lose = self.score[loser_id]["points"]

            # Win game: 4+ points with 2-point lead
            if p_win >= 4 and p_win - p_lose >= 2:
                self._award_game(winner_id)
                self.score[0]["points"] = 0
                self.score[1]["points"] = 0
                self.server = 1 - self.server  # Switch server after game

        # Record rally
        self.rally_history.append({
            "winner": winner_id,
            "reason": reason,
            "score": self.get_score_string(),
        })

    def _award_game(self, winner_id: int):
        """Award game to winner."""
        loser_id = 1 - winner_id
        self.score[winner_id]["games"] += 1

        g_win = self.score[winner_id]["games"]
        g_lose = self.score[loser_id]["games"]

        # Check for set win
        # Standard: 6 games with 2-game lead
        if g_win >= 6 and g_win - g_lose >= 2:
            self._award_set(winner_id)
        # Winning tiebreak at 7-6 (after 6-6 tiebreak)
        elif g_win == 7 and g_lose == 6:
            self._award_set(winner_id)
        # Tiebreak at 6-6
        elif g_win == 6 and g_lose == 6:
            self.tiebreak = True

    def _award_set(self, winner_id: int):
        """Award set to winner."""
        self.score[winner_id]["sets"] += 1
        self.score[0]["games"] = 0
        self.score[1]["games"] = 0

        # Check for match win
        sets_to_win = (self.best_of + 1) // 2  # 2 for best of 3, 3 for best of 5
        if self.score[winner_id]["sets"] >= sets_to_win:
            self.match_over = True

    def get_score_string(self) -> str:
        """Get formatted score string.

        Returns:
            Score string like "30-15 | Games: 2-1 | Sets: 1-0"
        """
        if self.tiebreak:
            # Tiebreak score
            return f"Tiebreak: {self.score[0]['points']}-{self.score[1]['points']}"

        # Point score
        point_map = {0: "0", 1: "15", 2: "30", 3: "40"}
        p0 = self.score[0]["points"]
        p1 = self.score[1]["points"]

        # Deuce/Advantage
        if p0 >= 3 and p1 >= 3:
            if p0 == p1:
                point_str = "Deuce"
            elif p0 > p1:
                point_str = f"Adv {self.player_names[0]}"
            else:
                point_str = f"Adv {self.player_names[1]}"
        else:
            point_str = f"{point_map.get(p0, '40')}-{point_map.get(p1, '40')}"

        # Game score
        game_str = f"{self.score[0]['games']}-{self.score[1]['games']}"

        # Set score
        set_str = f"Sets: {self.score[0]['sets']}-{self.score[1]['sets']}"

        return f"{point_str} | Games: {game_str} | {set_str}"

    def get_final_score(self) -> str:
        """Get final match score.

        Returns:
            Final score string like "6-4, 6-3" or "6-4, 4-6, 6-2"
        """
        # This would need to track set scores throughout the match
        # For now, return a simple representation
        return f"{self.score[0]['sets']}-{self.score[1]['sets']}"

    def get_winner(self) -> Optional[int]:
        """Get match winner.

        Returns:
            Winner player ID (0 or 1) or None if match not over
        """
        if not self.match_over:
            return None

        return 0 if self.score[0]["sets"] > self.score[1]["sets"] else 1

    def get_stats(self) -> dict:
        """Get match statistics.

        Returns:
            Dictionary with rallies, points won, current score, match status
        """
        return {
            "rallies": len(self.rally_history),
            "points_won": {
                0: self.stats[0]["points_won"],
                1: self.stats[1]["points_won"],
            },
            "current_score": self.get_score_string(),
            "match_over": self.match_over,
            "player_stats": self.stats,
        }

    def get_player_stats(self, player_id: int) -> dict:
        """Get statistics for a specific player.

        Args:
            player_id: Player ID (0 or 1)

        Returns:
            Player statistics dictionary
        """
        return self.stats[player_id].copy()
