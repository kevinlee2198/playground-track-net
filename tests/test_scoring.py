"""Tests for TennisScorer."""

import pytest

from inference.scoring import TennisScorer


def test_tennis_scorer_initialization():
    """Test scorer initializes correctly."""
    scorer = TennisScorer("Player 1", "Player 2")

    assert scorer.player_names == ["Player 1", "Player 2"]
    assert scorer.best_of == 3
    assert scorer.score[0]["sets"] == 0
    assert scorer.score[0]["games"] == 0
    assert scorer.score[0]["points"] == 0
    assert scorer.match_over is False


def test_tennis_scorer_custom_best_of():
    """Test scorer with best of 5 sets."""
    scorer = TennisScorer(best_of=5)

    assert scorer.best_of == 5


def test_award_single_point():
    """Test awarding a single point."""
    scorer = TennisScorer()

    scorer.award_point(0)

    assert scorer.score[0]["points"] == 1
    assert scorer.score[1]["points"] == 0


def test_point_scoring_sequence():
    """Test standard point scoring (0, 15, 30, 40)."""
    scorer = TennisScorer()

    # Player 0 wins 4 points
    scorer.award_point(0)  # 15-0
    assert scorer.score[0]["points"] == 1

    scorer.award_point(0)  # 30-0
    assert scorer.score[0]["points"] == 2

    scorer.award_point(0)  # 40-0
    assert scorer.score[0]["points"] == 3

    scorer.award_point(0)  # Game
    assert scorer.score[0]["games"] == 1
    assert scorer.score[0]["points"] == 0  # Points reset


def test_game_win_with_two_point_lead():
    """Test game win requires 2-point lead."""
    scorer = TennisScorer()

    # Get to 40-40 (deuce)
    for _ in range(3):
        scorer.award_point(0)
        scorer.award_point(1)

    assert scorer.score[0]["points"] == 3
    assert scorer.score[1]["points"] == 3

    # Player 0 gets advantage
    scorer.award_point(0)
    assert scorer.score[0]["points"] == 4
    assert scorer.score[0]["games"] == 0  # Not won yet

    # Back to deuce
    scorer.award_point(1)
    assert scorer.score[0]["points"] == 4
    assert scorer.score[1]["points"] == 4

    # Player 1 gets advantage and wins
    scorer.award_point(1)
    scorer.award_point(1)

    assert scorer.score[1]["games"] == 1
    assert scorer.score[0]["points"] == 0  # Points reset
    assert scorer.score[1]["points"] == 0


def test_set_win_at_six_games():
    """Test set win at 6 games with 2-game lead."""
    scorer = TennisScorer()

    # Player 0 wins 6 games quickly
    for _ in range(6):
        for _ in range(4):  # 4 points per game
            scorer.award_point(0)

    assert scorer.score[0]["sets"] == 1
    assert scorer.score[0]["games"] == 0  # Games reset


def test_tiebreak_at_six_six():
    """Test tiebreak triggered at 6-6."""
    scorer = TennisScorer()

    # Get to 6-6
    for _ in range(6):
        # Player 0 wins a game
        for _ in range(4):
            scorer.award_point(0)
        # Player 1 wins a game
        for _ in range(4):
            scorer.award_point(1)

    assert scorer.score[0]["games"] == 6
    assert scorer.score[1]["games"] == 6
    assert scorer.tiebreak is True


def test_tiebreak_scoring():
    """Test tiebreak point scoring."""
    scorer = TennisScorer()

    # Get to 6-6 (trigger tiebreak)
    for _ in range(6):
        for _ in range(4):
            scorer.award_point(0)
        for _ in range(4):
            scorer.award_point(1)

    assert scorer.tiebreak is True

    # Play tiebreak
    for _ in range(7):
        scorer.award_point(0)

    # Player 0 should win tiebreak (7-0)
    assert scorer.tiebreak is False
    assert scorer.score[0]["sets"] == 1
    assert scorer.score[0]["games"] == 0  # Reset after set win


def test_tiebreak_requires_two_point_lead():
    """Test tiebreak requires 2-point lead."""
    scorer = TennisScorer()

    # Get to 6-6
    for _ in range(6):
        for _ in range(4):
            scorer.award_point(0)
        for _ in range(4):
            scorer.award_point(1)

    # Play to 6-6 in tiebreak
    for _ in range(6):
        scorer.award_point(0)
        scorer.award_point(1)

    assert scorer.score[0]["points"] == 6
    assert scorer.score[1]["points"] == 6
    assert scorer.tiebreak is True  # Still in tiebreak

    # Player 0 gets 7-6 (not enough lead)
    scorer.award_point(0)
    assert scorer.tiebreak is True

    # Player 0 gets 8-6 (2-point lead, wins)
    scorer.award_point(0)
    assert scorer.tiebreak is False
    assert scorer.score[0]["sets"] == 1


def test_match_win_best_of_three():
    """Test match win in best of 3 sets."""
    scorer = TennisScorer(best_of=3)

    # Player 0 wins 2 sets
    for set_num in range(2):
        for _ in range(6):
            for _ in range(4):
                scorer.award_point(0)

    assert scorer.match_over is True
    assert scorer.score[0]["sets"] == 2


def test_match_win_best_of_five():
    """Test match win in best of 5 sets."""
    scorer = TennisScorer(best_of=5)

    # Player 0 wins 3 sets
    for set_num in range(3):
        for _ in range(6):
            for _ in range(4):
                scorer.award_point(0)

    assert scorer.match_over is True
    assert scorer.score[0]["sets"] == 3


def test_no_points_after_match_over():
    """Test points cannot be awarded after match ends."""
    scorer = TennisScorer()

    # Win match quickly
    for _ in range(2):  # 2 sets
        for _ in range(6):  # 6 games
            for _ in range(4):  # 4 points
                scorer.award_point(0)

    assert scorer.match_over is True

    # Try to award more points
    scorer.award_point(1)

    # Score shouldn't change
    assert scorer.score[1]["points"] == 0


def test_get_score_string_basic():
    """Test score string formatting."""
    scorer = TennisScorer()

    # 0-0
    assert "0-0" in scorer.get_score_string()

    # 15-0
    scorer.award_point(0)
    assert "15-0" in scorer.get_score_string()

    # 15-15
    scorer.award_point(1)
    assert "15-15" in scorer.get_score_string()


def test_get_score_string_deuce():
    """Test score string shows deuce."""
    scorer = TennisScorer()

    # Get to deuce (40-40)
    for _ in range(3):
        scorer.award_point(0)
        scorer.award_point(1)

    assert "Deuce" in scorer.get_score_string()


def test_get_score_string_advantage():
    """Test score string shows advantage."""
    scorer = TennisScorer("Alice", "Bob")

    # Get to deuce
    for _ in range(3):
        scorer.award_point(0)
        scorer.award_point(1)

    # Alice gets advantage
    scorer.award_point(0)
    score_str = scorer.get_score_string()
    assert "Adv" in score_str
    assert "Alice" in score_str


def test_get_score_string_tiebreak():
    """Test score string during tiebreak."""
    scorer = TennisScorer()

    # Get to 6-6
    for _ in range(6):
        for _ in range(4):
            scorer.award_point(0)
        for _ in range(4):
            scorer.award_point(1)

    # In tiebreak
    scorer.award_point(0)
    scorer.award_point(0)
    scorer.award_point(1)

    score_str = scorer.get_score_string()
    assert "Tiebreak" in score_str
    assert "2-1" in score_str


def test_server_switches_after_game():
    """Test server switches after each game."""
    scorer = TennisScorer()

    assert scorer.server == 0

    # Player 0 wins game
    for _ in range(4):
        scorer.award_point(0)

    assert scorer.server == 1


def test_stats_tracking_aces():
    """Test ace statistics tracking."""
    scorer = TennisScorer()

    scorer.award_point(0, reason="ace")
    scorer.award_point(0, reason="ace")

    assert scorer.stats[0]["aces"] == 2


def test_stats_tracking_winners():
    """Test winner statistics tracking."""
    scorer = TennisScorer()

    scorer.award_point(0, reason="winner")
    scorer.award_point(1, reason="winner")

    assert scorer.stats[0]["winners"] == 1
    assert scorer.stats[1]["winners"] == 1


def test_stats_tracking_unforced_errors():
    """Test unforced error tracking."""
    scorer = TennisScorer()

    # Player 0 wins because player 1 made error
    scorer.award_point(0, reason="out")

    assert scorer.stats[1]["unforced_errors"] == 1


def test_stats_tracking_double_faults():
    """Test double fault tracking."""
    scorer = TennisScorer()

    scorer.award_point(0, reason="double_fault")

    assert scorer.stats[1]["double_faults"] == 1


def test_stats_points_won():
    """Test points won tracking."""
    scorer = TennisScorer()

    scorer.award_point(0)
    scorer.award_point(0)
    scorer.award_point(1)

    assert scorer.stats[0]["points_won"] == 2
    assert scorer.stats[1]["points_won"] == 1


def test_rally_history():
    """Test rally history is recorded."""
    scorer = TennisScorer()

    scorer.award_point(0, reason="ace")
    scorer.award_point(1, reason="winner")

    assert len(scorer.rally_history) == 2
    assert scorer.rally_history[0]["winner"] == 0
    assert scorer.rally_history[0]["reason"] == "ace"
    assert scorer.rally_history[1]["winner"] == 1
    assert scorer.rally_history[1]["reason"] == "winner"


def test_get_winner_during_match():
    """Test get_winner returns None during match."""
    scorer = TennisScorer()

    scorer.award_point(0)

    assert scorer.get_winner() is None


def test_get_winner_after_match():
    """Test get_winner returns winner after match."""
    scorer = TennisScorer()

    # Player 0 wins 2 sets
    for _ in range(2):
        for _ in range(6):
            for _ in range(4):
                scorer.award_point(0)

    assert scorer.get_winner() == 0


def test_reset_match():
    """Test match can be reset."""
    scorer = TennisScorer()

    # Play some points
    scorer.award_point(0)
    scorer.award_point(1)

    # Reset
    scorer.reset_match()

    assert scorer.score[0]["points"] == 0
    assert scorer.score[1]["points"] == 0
    assert scorer.match_over is False
    assert len(scorer.rally_history) == 0


def test_get_stats():
    """Test get_stats returns correct data."""
    scorer = TennisScorer()

    scorer.award_point(0)
    scorer.award_point(1)

    stats = scorer.get_stats()

    assert stats["rallies"] == 2
    assert stats["points_won"][0] == 1
    assert stats["points_won"][1] == 1
    assert stats["match_over"] is False


def test_get_player_stats():
    """Test get_player_stats returns individual player stats."""
    scorer = TennisScorer()

    scorer.award_point(0, reason="ace")
    scorer.award_point(0, reason="winner")

    p0_stats = scorer.get_player_stats(0)

    assert p0_stats["points_won"] == 2
    assert p0_stats["aces"] == 1
    assert p0_stats["winners"] == 1
