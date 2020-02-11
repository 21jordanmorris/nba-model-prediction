# How I want to build my model:
#   * Determine which stats are most impactful/important
#   * Train a model using just that
#   * After finishing that, add weight to previous 5-15
#   * If that goes well, then try and add injury weights

from basketball_reference_scraper.teams import get_roster, get_team_stats, get_opp_stats, get_roster_stats, get_team_misc
from basketball_reference_scraper.seasons import get_schedule
from basketball_reference_scraper.box_scores import get_box_scores

entire_schedule = get_schedule(2020, playoffs=False)
past_schedule = entire_schedule.dropna()

num_games = past_schedule.shape[0]
num_homewins = len(past_schedule[past_schedule.HOME_PTS > past_schedule.VISITOR_PTS])
home_team_win_rate = (num_homewins / num_games) * 100
away_team_win_rate = 100 - home_team_win_rate

pelicans = get_team_stats('NOP', 2020)

# Add Winner Column to the Past Schedule Dataframe
winner = []
for index, row in past_schedule.iterrows():
    if row['HOME_PTS'] > row['VISITOR_PTS']:
        winner.insert(index, 'H')
    else:
        winner.insert(index, 'V')

past_schedule.insert(5, "Winner", winner, True)

print(past_schedule)