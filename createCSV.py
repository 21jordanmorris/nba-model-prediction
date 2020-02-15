from basketball_reference_scraper.teams import get_roster, get_team_stats, get_opp_stats, get_roster_stats, get_team_misc
from basketball_reference_scraper.seasons import get_schedule
from basketball_reference_scraper.box_scores import get_box_scores
from utils import *
import sys
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None

def convert_team_names(entire_schedule):
    team_abbreviations = {'Atlanta Hawks' : 'ATL', 
                        'Brooklyn Nets' : 'BRK', 
                        'Boston Celtics' : 'BOS',
                        'Charlotte Hornets' : 'CHO',
                        'Chicago Bulls' : 'CHI',
                        'Cleveland Cavaliers' : 'CLE',
                        'Dallas Mavericks' : 'DAL',
                        'Denver Nuggets' : 'DEN',
                        'Detroit Pistons' : 'DET',
                        'Golden State Warriors' : 'GSW',
                        'Houston Rockets' : 'HOU',
                        'Indiana Pacers' : 'IND',
                        'Los Angeles Clippers' : 'LAC',
                        'Los Angeles Lakers' : 'LAL', 
                        'Memphis Grizzlies' : 'MEM', 
                        'Miami Heat' : 'MIA', 
                        'Milwaukee Bucks' : 'MIL', 
                        'Minnesota Timberwolves' : 'MIN', 
                        'New Orleans Pelicans' : 'NOP', 
                        'New York Knicks' : 'NYK', 
                        'Oklahoma City Thunder' : 'OKC', 
                        'Orlando Magic' : 'ORL', 
                        'Philadelphia 76ers' : 'PHI', 
                        'Phoenix Suns' : 'PHO', 
                        'Portland Trail Blazers' : 'POR', 
                        'Sacramento Kings' : 'SAC', 
                        'San Antonio Spurs' : 'SAS', 
                        'Toronto Raptors' : 'TOR', 
                        'Utah Jazz' : 'UTA', 
                        'Washington Wizards' : 'WAS'}

    team_abbreviations_keys = list(team_abbreviations.keys())

    row = 0
    for home_team in entire_schedule['HOME']:
        for i in range(0, 30):
            if team_abbreviations_keys[i] in home_team:
                entire_schedule.at[row, 'HOME'] = team_abbreviations[team_abbreviations_keys[i]]
        row += 1
    row = 0
    for away_team in entire_schedule['VISITOR']:
        for i in range(0, 30):
            if team_abbreviations_keys[i] in away_team:
                entire_schedule.at[row, 'VISITOR'] = team_abbreviations[team_abbreviations_keys[i]]
        row += 1
    
    return entire_schedule

def add_winner_column(entire_schedule):
    winner = []
    for index, row in entire_schedule.iterrows():
        if np.isnan(row['HOME_PTS']) or np.isnan(row['VISITOR_PTS']):
            winner.insert(index, float('Nan'))
        elif row['HOME_PTS'] > row['VISITOR_PTS']:
            winner.insert(index, 1)
        else:
            winner.insert(index, 0)
    entire_schedule["WINNER"] = winner
    return entire_schedule

def add_team_stats(entire_schedule):
    team_misc_2020 = getTeamMisc(2020)
    team_misc_2019 = getTeamMisc(2019)
    last_year_win_percentage_home = []
    last_year_win_percentage_visitor = []
    count = 1
    for index, row in entire_schedule.iterrows():
        home = team_misc_2019.loc[team_misc_2019['TEAM'] == row['HOME'], 'W'].values[0]
        visitor = team_misc_2019.loc[team_misc_2019['TEAM'] == row['VISITOR'], 'W'].values[0]

        last_year_win_percentage_home.insert(index, home/82)
        last_year_win_percentage_visitor.insert(index, visitor/82)

        home_team_season_stats = team_misc_2020[team_misc_2020['TEAM'] == row['HOME']].add_prefix("HOME_")
        visitor_team_season_stats = team_misc_2020[team_misc_2020['TEAM'] == row['VISITOR']].add_prefix("VISITOR_")

        home_team_season_stats = home_team_season_stats[['HOME_NRtg', 'HOME_DRB%', 'HOME_SRS']]
        visitor_team_season_stats = visitor_team_season_stats[['VISITOR_NRtg', 'VISITOR_DRB%', 'VISITOR_SRS']]
        
        if count == 1:
            home_team_stats = home_team_season_stats
            visitor_team_stats = visitor_team_season_stats
        else:
            home_team_stats = pd.concat([home_team_stats, home_team_season_stats])
            visitor_team_stats = pd.concat([visitor_team_stats, visitor_team_season_stats])
        count += 1

    entire_schedule = entire_schedule.reset_index(drop=True)
    home_team_stats = home_team_stats.reset_index(drop=True)
    visitor_team_stats = visitor_team_stats.reset_index(drop=True)
    entire_schedule = pd.concat([entire_schedule, home_team_stats, visitor_team_stats], axis=1, sort=True)
    entire_schedule["HOME_LAST_SEASON_W%"] = last_year_win_percentage_home
    entire_schedule["VISITOR_LAST_SEASON_W%"] = last_year_win_percentage_visitor
    entire_schedule["HOME_GAME"] = [1] * len(entire_schedule)
    entire_schedule["AWAY GAME"] = [0] * len(entire_schedule)
    return entire_schedule

def add_win_percentage(entire_schedule):
    teams = [
        'ATL','BRK','BOS','CHO','CHI','CLE','DAL','DEN',
        'DET','GSW','HOU','IND','LAC','LAL','MEM','MIA', 
        'MIL','MIN','NOP','NYK','OKC','ORL','PHI','PHO', 
        'POR','SAC','SAS','TOR','UTA','WAS']
    init = [0] * 30
    data = {'wins' : init, 'losses' : init}

    win_loss = pd.DataFrame(data, index=teams)
    home_win_percentage = []
    visitor_win_percentage = []
    for index, row in entire_schedule.iterrows():
        
        if (win_loss.loc[row['HOME'], 'wins'] + win_loss.loc[row['HOME'], 'losses']) == 0:
            if win_loss.loc[row['HOME'], 'wins'] > 0:
                home_win_per = 1
            else:
                home_win_per = 0
        else:
            home_wins = win_loss.loc[row['HOME'], 'wins']
            home_win_per = home_wins / (win_loss.loc[row['HOME'], 'wins'] + win_loss.loc[row['HOME'], 'losses'])

        if (win_loss.loc[row['VISITOR'], 'wins'] + win_loss.loc[row['VISITOR'], 'losses']) == 0:
            if win_loss.loc[row['VISITOR'], 'wins'] > 0:
                visitor_win_per = 1
            else:
                visitor_win_per = 0
        else:
            visitor_wins = win_loss.loc[row['VISITOR'], 'wins']
            visitor_win_per = visitor_wins / (win_loss.loc[row['VISITOR'], 'wins'] + win_loss.loc[row['VISITOR'], 'losses'])

        home_win_percentage.insert(index, home_win_per)
        visitor_win_percentage.insert(index, visitor_win_per)

        if row['WINNER'] == 1:
            win_loss.at[row['HOME'],'wins'] = win_loss.loc[row['HOME'],'wins'] + 1
            win_loss.at[row['VISITOR'],'losses'] = win_loss.loc[row['VISITOR'],'losses'] + 1
        elif row['WINNER'] == 0:
            win_loss.at[row['VISITOR'],'wins'] = win_loss.loc[row['VISITOR'],'wins'] + 1
            win_loss.at[row['HOME'],'losses'] = win_loss.loc[row['HOME'],'losses'] + 1

    entire_schedule["HOME_W%"] = home_win_percentage
    entire_schedule["VISITOR_W%"] = visitor_win_percentage
    return entire_schedule

def add_home_away_splits(entire_schedule):
    teams = [
        'ATL','BRK','BOS','CHO','CHI','CLE','DAL','DEN',
        'DET','GSW','HOU','IND','LAC','LAL','MEM','MIA', 
        'MIL','MIN','NOP','NYK','OKC','ORL','PHI','PHO', 
        'POR','SAC','SAS','TOR','UTA','WAS']
    init = [0] * 30
    data = {'home_wins' : init, 'home_losses' : init, 'road_wins': init, 'road_losses' : init}
    home_road_split = pd.DataFrame(data, index=teams)
    home_win_at_home = []
    visitor_win_on_road = []
    for index, row in entire_schedule.iterrows():
        if (home_road_split.loc[row['HOME'], 'home_wins'] + home_road_split.loc[row['HOME'], 'home_losses']) == 0:
            if home_road_split.loc[row['HOME'], 'home_wins'] > 0:
                home_win_per = 1
            else:
                home_win_per = 0
        else:
            home_wins = home_road_split.loc[row['HOME'], 'home_wins']
            home_win_per = home_wins / (home_road_split.loc[row['HOME'], 'home_wins'] + home_road_split.loc[row['HOME'], 'home_losses'])

        if (home_road_split.loc[row['VISITOR'], 'road_wins'] + home_road_split.loc[row['VISITOR'], 'road_losses']) == 0:
            if home_road_split.loc[row['VISITOR'], 'road_wins'] > 0:
                visitor_win_per = 1
            else:
                visitor_win_per = 0
        else:
            visitor_wins = home_road_split.loc[row['VISITOR'], 'road_wins']
            visitor_win_per = visitor_wins / (home_road_split.loc[row['VISITOR'], 'road_wins'] + home_road_split.loc[row['VISITOR'], 'road_losses'])

        home_win_at_home.insert(index, home_win_per)
        visitor_win_on_road.insert(index, visitor_win_per)

        if row['WINNER'] == 1:
            home_road_split.at[row['HOME'],'home_wins'] = home_road_split.loc[row['HOME'],'home_wins'] + 1
            home_road_split.at[row['VISITOR'],'road_losses'] = home_road_split.loc[row['VISITOR'],'road_losses'] + 1
        elif row['WINNER'] == 0:
            home_road_split.at[row['VISITOR'],'road_wins'] = home_road_split.loc[row['VISITOR'],'road_wins'] + 1
            home_road_split.at[row['HOME'],'home_losses'] = home_road_split.loc[row['HOME'],'home_losses'] + 1

    entire_schedule['HOME_W%_AT_HOME'] = home_win_at_home
    entire_schedule['VISITOR_W%_ON_ROAD'] = visitor_win_on_road
    return entire_schedule

def add_second_of_b2b(entire_schedule):
    teams_that_played_last_night = []
    teams_that_played_today = []
    home_team_b2b = []
    visitor_team_b2b = []
    last_night_day = "Monday"

    for index, row in entire_schedule.iterrows():
        day = pd.Timestamp.day_name(row['DATE'])
        
        if get_next_day(last_night_day) != day:
            teams_that_played_last_night = teams_that_played_today
            teams_that_played_today = []
            last_night_day = get_day_before(day)
        
        if get_next_day(last_night_day) == day:
            if row['HOME'] in teams_that_played_last_night:
                home_team_b2b.insert(index, 1)
            else:
                home_team_b2b.insert(index, 0)

            if row['VISITOR'] in teams_that_played_last_night:
                visitor_team_b2b.insert(index, 1)
            else:
                visitor_team_b2b.insert(index, 0)
        
        teams_that_played_today.append(row['HOME'])
        teams_that_played_today.append(row['VISITOR'])

    entire_schedule["HOME_B2B"] = home_team_b2b
    entire_schedule["VISITOR_B2B"] = visitor_team_b2b
    return entire_schedule

def add_last_five(entire_schedule):
    team_last_five = get_teams_last_five_init()
    home_last_five_percent = []
    visitor_last_five_percent = []
    for index, row in entire_schedule.iterrows():

        home_last_five_percent.append(get_win_loss_percentage_last_five(row['HOME'], team_last_five))
        visitor_last_five_percent.append(get_win_loss_percentage_last_five(row['VISITOR'], team_last_five))
        
        if not np.isnan(row['WINNER']):
            temp_home = team_last_five[row['HOME']]
            temp_home.pop(0)
            temp_visitor = team_last_five[row['VISITOR']]
            temp_visitor.pop(0)

            if row['WINNER'] == 1:
                temp_home.append('W')
                temp_visitor.append('L')
                team_last_five.update({row['HOME'] : temp_home})
                team_last_five.update({row['VISITOR'] : temp_visitor})
            elif row['WINNER'] == 0:
                temp_home.append('L')
                temp_visitor.append('W')
                team_last_five.update({row['HOME'] : temp_home})
                team_last_five.update({row['VISITOR'] : temp_visitor})

    entire_schedule["HOME_LAST_5"] = home_last_five_percent
    entire_schedule["VISITOR_LAST_5"] = visitor_last_five_percent
    return entire_schedule

def finalize_csv():
    entire_schedule = get_schedule(2020, playoffs=False)
    convert_team_names(entire_schedule)
    add_winner_column(entire_schedule)
    entire_schedule = add_win_percentage(add_team_stats(entire_schedule))
    entire_schedule = add_second_of_b2b(entire_schedule)
    entire_schedule = add_home_away_splits(entire_schedule)
    entire_schedule = add_last_five(entire_schedule)
    entire_schedule.to_csv('season_2020.csv', index=False)
    
    sys.stderr.write("[PROGRESS] CSV file creation completed!")
    sys.stderr.flush()
    
    return "season_2020.csv"