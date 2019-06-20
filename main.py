from bs4 import BeautifulSoup, Comment
import requests
import csv
import numpy as np
from sklearn import linear_model
import time
from pybaseball import pitching_stats, batting_stats
import pandas as pd
from datetime import date
import pickle
import os.path
from os import path

battingStats = [
    'batters_used',
    'age_bat',
    'runs_per_game',
    'G',
    'PA',
    'AB',
    'R',
    'H',
    '2B',
    '3B',
    'HR',
    'RBI',
    'SB',
    'CS',
    'BB',
    'SO',
    'batting_avg',
    'onbase_perc',
    'slugging_perc',
    'onbase_plus_slugging',
    'onbase_plus_slugging_plus',
    'TB',
    'GIDP',
    'HBP',
    'SH',
    'SF',
    'IBB',
    'LOB'
]

pitchingStats = [
    'pitchers_used',
    'age_pitch',
    'runs_allowed_per_game',
    'W',
    'L',
    'win_loss_perc',
    'earned_run_avg',
    'G',
    'GS',
    'GF',
    'CG',
    'SHO_team',
    'SHO_cg',
    'SV',
    'IP',
    'H',
    'R',
    'ER',
    'HR',
    'BB',
    'IBB',
    'SO',
    'HBP',
    'BK',
    'WP',
    'batters_faced',
    'earned_run_avg_plus',
    'fip',
    'whip',
    'hits_per_nine',
    'home_runs_per_nine',
    'bases_on_balls_per_nine',
    'strikeouts_per_nine',
    'strikeouts_per_base_on_balls',
    'LOB'
]

fangraphsProjBattingStats = [
    "Name",
    "info",
    "Team",
    "G",
    "PA",
    "AB",
    "H",
    "2B",
    "3B",
    "HR",
    "R",
    "RBI",
    "BB",
    "SO",
    "HBP",
    "SB",
    "CS",
    "AVG",
    "OBP",
    "SLG",
    "OPS",
    "wOBA",
    "Fld",
    "BsR",
    "WAR"
]

fangraphsProjPitchingStats = [
    "Name",
    "info",
    "Team",
    "W",
    "L",
    "SV",
    "HLD",
    "ERA",
    "GS",
    "G",
    "IP",
    "H",
    "ER",
    "HR",
    "SO",
    "BB",
    "WHIP",
    "K/9",
    "BB/9",
    "FIP",
    "WAR"
]

fangraphsToTeamDict = {
    "Los Angeles Angels of Anaheim": 1,
    "Los Angeles Angels": 1,
    "Houston Astros": 21,
    "Oakland Athletics": 10,
    "Toronto Blue Jays": 14,
    "Atlanta Braves": 16,
    "Milwaukee Brewers": 23,
    "St. Louis Cardinals": 28,
    "Chicago Cubs": 17,
    "Arizona Diamondbacks": 15,
    "Los Angeles Dodgers": 22,
    "San Francisco Giants": 30,
    "Cleveland Indians": 5,
    "Seattle Mariners": 11,
    "Miami Marlins": 20,
    "New York Mets": 25,
    "Washington Nationals": 24,
    "Baltimore Orioles": 2,
    "San Diego Padres": 29,
    "Texas Rangers": 13,
    "Tampa Bay Rays": 12,
    "Boston Red Sox": 3,
    "Cincinnati Reds": 18,
    "Colorado Rockies": 19,
    "Kansas City Royals": 7,
    "Detroit Tigers": 6,
    "Minnesota Twins": 8,
    "Chicago White Sox": 4,
    "New York Yankees": 9,
    "Pittsburgh Pirates": 27,
    "Philadelphia Phillies": 26
}

bbrefToFangraphsDict = {
    "Los Angeles Angels of Anaheim": "Angels",
    "Los Angeles Angels": "Angels",
    "Houston Astros": "Astros",
    "Oakland Athletics": "Athletics",
    "Toronto Blue Jays": "Blue Jays",
    "Atlanta Braves": "Braves",
    "Milwaukee Brewers": "Brewers",
    "St. Louis Cardinals": "Cardinals",
    "Chicago Cubs": "Cubs",
    "Arizona Diamondbacks": "Diamondbacks",
    "Los Angeles Dodgers": "Dodgers",
    "San Francisco Giants": "Giants",
    "Cleveland Indians": "Indians",
    "Seattle Mariners": "Mariners",
    "Miami Marlins": "Marlins",
    "New York Mets": "Mets",
    "Washington Nationals": "Nationals",
    "Baltimore Orioles": "Orioles",
    "San Diego Padres": "Padres",
    "Texas Rangers": "Rangers",
    "Tampa Bay Rays": "Rays",
    "Boston Red Sox": "Red Sox",
    "Cincinnati Reds": "Reds",
    "Colorado Rockies": "Rockies",
    "Kansas City Royals": "Royals",
    "Detroit Tigers": "Tigers",
    "Minnesota Twins": "Twins",
    "Chicago White Sox": "White Sox",
    "New York Yankees": "Yankees",
    "Pittsburgh Pirates": "Pirates",
    "Philadelphia Phillies": "Phillies"
}

abbreviationToFangraphsDict = {
    "LAA": "Angels",
    "HOU": "Astros",
    "OAK": "Athletics",
    "TOR": "Blue Jays",
    "ATL": "Braves",
    "MIL": "Brewers",
    "STL": "Cardinals",
    "CHC": "Cubs",
    "ARI": "Diamondbacks",
    "LAD": "Dodgers",
    "SF": "Giants",
    "CLE": "Indians",
    "SEA": "Mariners",
    "MIA": "Marlins",
    "NYM": "Mets",
    "WSH": "Nationals",
    "WAS": "Nationals",
    "BAL": "Orioles",
    "SD": "Padres",
    "TEX": "Rangers",
    "TB": "Rays",
    "BOS": "Red Sox",
    "CIN": "Reds",
    "COL": "Rockies",
    "KC": "Royals",
    "DET": "Tigers",
    "MIN": "Twins",
    "CWS": "White Sox",
    "NYY": "Yankees",
    "PIT": "Pirates",
    "PHI": "Phillies"
}


MASTER_SEASONS = [2013, 2014, 2015, 2016, 2017, 2018, 2019]
GAMES_IN_SEASON = 162
R_W = 10.119

class Season:
    def __init__(self, season, teams):
        self.season = season
        self.teams = teams

    def addwOBAFIP(self, constants):
        self.wOBAFIPConstants = constants

    def addTeam(self, team):
        self.teams.append(team)

    def battingTotals(self, total):
        self.battingTotal = total

    def pitchingTotals(self, total):
        self.pitchingTotal = total

    def getTeam(self, fullName):
        for team in self.teams:
            if team.team == fullName:
                return team

class BaseballTeam:
    def __init__(self, name, abbr):
        self.team = name
        self.abbreviation = abbr
        self.fgTeamName = ""
        self.projWinPercentageYear = 0
        self.projWinsYear = 0
        self.projLosesYear = 0

    def addPitching(self, pitching):
        self.pitching = pitching

    def addBatting(self, batting):
        self.batting = batting

    def getBatting(self):
        return self.batting

    def getPitching(self):
        return self.pitching

class BattingTeam:
    def __init__(self, numBatters, avgBatterAge, runsPerGame, games, totPlateAppearances, totAtBats, totRuns, totHits, totDoubles,
                 totTriples, totHRs, totRBIs, totSBs, totCSs, totBBs, totSOs, avgBAs, avgOBPs, avgSLG, avgOPS, avgOPSPlus, totBases,
                 totGDP, totHBP, totSH, totSF, totIBB, totLOB):
        self.numBatters = numBatters
        self.avgBatterAge = avgBatterAge
        self.runsPerGame = runsPerGame
        self.games = games
        self.totPlateAppearances = totPlateAppearances
        self.totAtBats = totAtBats
        self.totRuns = totRuns
        self.totHits = totHits
        self.totDoubles = totDoubles
        self.totTriples = totTriples
        self.totHRs = totHRs
        self.totRBIs = totRBIs
        self.totSBs = totSBs
        self.totCSs = totCSs
        self.totBBs = totBBs
        self.totSOs = totSOs
        self.avgBAs = avgBAs
        self.avgOBPs = avgOBPs
        self.avgOPSPlus = avgOPSPlus
        self.avgSLG = avgSLG
        self.avgOPS = avgOPS
        self.totBases = totBases
        self.totGDP = totGDP
        self.totHBP = totHBP
        self.totSH = totSH
        self.totSF = totSF
        self.totIBB = totIBB
        self.totLOB = totLOB
        self.batters = []
        self.luck = 0
        self.projectedRuns = 0
        self.batterWar = 0
        self.changeInBattingWarFromPrevYear = 0
        self.changeInBattingRunsFromPrevYear = 0
        self.projectedRSYear = 0
        self.projectedRSPercentYear = 0

    def addWOBA(self, wOBA):
        self.wOBA = wOBA

    def addBSR(self, bsr):
        self.BSR = bsr

    def addHitsPerRun(self, hitsPerRun):
        self.hitsPerRun = hitsPerRun

    def addISO(self, iso):
        self.iso = iso

    def addKKRate(self, kkRate):
        self.kRate = kkRate

    def addBBRate(self, bbRate):
        self.bbRate = bbRate

    def addBABIP (self, babip):
        self.babip = babip

    def addBatters(self, batters):
        self.batters = batters

    def setClusterLuck(self, luck):
        self.luck = luck

    def setProjectedRuns(self, runs):
        self.projectedRuns = runs

class PitchingTeam:
    def __init__(self, numPitchers, avgPitcherAge, runsAllowedGames, wins, losses, winLossPercentage, avgERA, games,
                 gamesStarted, gamesFinished, completedGames, totSho, comSho, save, inningsPitched, hits, runs, earnedRuns,
                 hrs, walks, ibbs, strikeouts, hbps, bks, wps, bf, eraPlus, fip, whip, hr9, bb9, so9, sow, lob):
        self.numPitchers = numPitchers
        self.avgPitcherAge = avgPitcherAge
        self.runsAllowedGames = runsAllowedGames
        self.wins = wins
        self.losses = losses
        self.winLossPercentage = winLossPercentage
        self.avgERA = avgERA
        self.games = games
        self.gamesStarted = gamesStarted
        self.gamesFinished = gamesFinished
        self.completedGames = completedGames
        self.totSho = totSho
        self.comSho = comSho
        self.save = save
        self.inningsPitched = inningsPitched
        self.hits = hits
        self.runs = runs
        self.earnedRuns = earnedRuns
        self.hrs = hrs
        self.walks = walks
        self.ibbs = ibbs
        self.strikeouts = strikeouts
        self.hbps = hbps
        self.bks = bks
        self.wps = wps
        self.bf = bf
        self.eraPlus = eraPlus
        self.fip = fip
        self.whip = whip
        self.hr9 = hr9
        self.bb9 = bb9
        self.so9 = so9
        self.sow = sow
        self.lob = lob
        self.pitchers = []
        self.relievers = []
        self.luck = 0
        self.projectedRuns = 0
        self.pitcherWar = 0
        self.changeInPitchingWarFromPrevYear = 0
        self.changeInPitchingRunsFromPrevYear = 0
        self.relieverWar = 0
        self.changeInReliefWarFromPrevYear = 0
        self.changeInReliefRunsFromPrevYear = 0
        self.projectedRAYear = 0
        self.projectedRAPercentYear = 0

    def addPitchers(self, pitchers):
        self.pitchers = pitchers

    def addRelievers(self, relievers):
        self.relievers = relievers

    def setClusterLuck(self, luck):
        self.luck = luck

    def setProjectedRuns(self, runs):
        self.projectedRuns = runs

class wOBAFIPConstants:
    def __init__(self, wOBA, wOBAScale, wBB, wHBP, wIB, w2B, w3B, wHR, runSB, runCS, RPerPA, RPerWin, cFIP):
        self.wOBA = wOBA
        self.wOBAScale = wOBAScale
        self.wBB = wBB
        self.wHBP = wHBP
        self.wIB = wIB
        self.w2B = w2B
        self.w3B = w3B
        self.wHR = wHR
        self.runSB = runSB
        self.runCS = runCS
        self.RPerPA = RPerPA
        self.RPerWin = RPerWin
        self.cFIP = cFIP

class Player:
    def __init__(self, playerName, id, team):
        self.playerName = playerName
        self.id = id
        self.team = team
        self.projection = {}
        self.seasonStats = {}
        self.pos = ""
        self.dailyAdjustedWAR = 0

    def setProjections(self, projections):
        self.projection = projections

    def setSeasonStats(self, seasonStats):
        self.seasonStats = seasonStats

    def setPos(self, pos):
        self.pos = pos

    def setDailyAdjWar(self, war):
        self.dailyAdjustedWAR = war

class Game:
    def __init__(self):
        self.teams = []
        self.totRunsScored = 0
        self.totRunsAllowed = 0
        self.runNormalization = 0

    def addTeam(self, team):
        self.teams.append(team)

class Lineup:
    def __init__(self, pitcher, home, away, teamName, opener):
        self.teamName = teamName
        self.home = home
        self.away = away
        self.opener = opener
        self.startingPitcher = pitcher
        self.adjustedBatterWar = 0
        self.changeBatterWar = 0
        self.adjustedPitchingWar = 0
        self.changePitchingWar = 0
        self.adjustedRelieverWar = 0
        self.projectedRS = 0
        self.projectedRSPercent = 0
        self.projectedRA = 0
        self.projectedRAPercent = 0
        self.winningPercentage = 0
        self.batters = []

    def addBatters(self, batters):
        self.batters = batters


def retrieveIDHelper(mappings, secondaryMappings, idIn, name, stringID, stringName):
    id = idIn

    try:
        id = mappings.loc[mappings[stringID] == str(int(id))].iloc[0]['mlb_id']
    except:
        try:
            id = mappings.loc[mappings['mlb_name'] == str(name)].iloc[0]['mlb_id']
        except:
            try:
                id = secondaryMappings.loc[secondaryMappings[stringID] == str(int(id))].iloc[0]['mlb_id']
            except:
                try:
                    id = secondaryMappings.loc[secondaryMappings['mlb_name'] == str(name)].iloc[0]['mlb_id']
                except:
                    print("id not found for: " + str(name) + " with id: " + str(idIn))
    return id

def get_lineups():
    mappings = getBaseballIDMappings()
    secondaryMappings = getSecondaryBaseballIDMappings()

    base_url = 'https://www.rotowire.com/baseball/daily-lineups.php'
    r = requests.get(base_url)

    soup = BeautifulSoup(r.text, 'html.parser')

    lineupClassName = "lineup"
    badLineupClass = "is-tools"
    badLineupClassTwo = "is-ad"

    lineupTeams = "lineup__team"

    pitcherInfoClass = "lineup__player-highlight-name"

    battersInfoClass = "lineup__list"

    player = "lineup__player"

    lineupClasses = soup.findAll("div", {"class": [lineupClassName]})

    games = []

    for lineup in lineupClasses:
        if badLineupClass not in lineup.attrs['class'] and badLineupClassTwo not in lineup.attrs['class']:
            teams = lineup.findAll("div", {"class": lineupTeams})

            awayTeamAbbr = teams[0].find("div", {"class": "lineup__abbr"}).text.strip()
            homeTeamAbbr = teams[1].find("div", {"class": "lineup__abbr"}).text.strip()
            awayTeamName = abbreviationToFangraphsDict[awayTeamAbbr]
            homeTeamName = abbreviationToFangraphsDict[homeTeamAbbr]

            print("Scraping lineup for team: " + str(awayTeamName))
            print("Scraping lineup for team: " + str(homeTeamName))

            teamNamesArray = [awayTeamName, homeTeamName]

            game = Game()

            pitchersInfo = lineup.findAll("div", {"class": pitcherInfoClass})

            awayTeam = None
            homeTeam = None

            count = 0
            pitchersFound = False

            for pitch in pitchersInfo:
                junk, rotowireID = pitch.a['href'].split("id=")
                name = pitch.a.text.strip()
                id = retrieveIDHelper(mappings, secondaryMappings, rotowireID, name, "rotowire_id", "rotowire_name")

                if count == 0:
                    play = Player(name, str(id), teamNamesArray[count])
                    print("Scraped pitcher: " + str(name) + " for team: " + str(teamNamesArray[count]))
                    awayTeam = Lineup(play, False, True, awayTeamName, None)
                elif count == 1:
                    play = Player(name, str(id), teamNamesArray[count])
                    print("Scraped pitcher: " + str(name) + " for team: " + str(teamNamesArray[count]))
                    homeTeam = Lineup(play, True, False, homeTeamName, None)
                    pitchersFound = True
                count += 1


            if pitchersFound and awayTeam is not None and homeTeam is not None:
                battersInfo = lineup.findAll("ul", {"class": battersInfoClass})

                batterCounter = 0
                for batLineup in battersInfo:
                    print("Found batters for: " + str(teamNamesArray[batterCounter]))
                    batters = batLineup.findAll("li", {"class": player})

                    batterPlayerData = []
                    for bat in batters:
                        name = bat.a['title'].strip()
                        junk, rotowireID = bat.a['href'].split("id=")
                        orderStancePos = bat.span.text.strip()
                        pos = bat.div.text.strip()

                        id = retrieveIDHelper(mappings, secondaryMappings, rotowireID, name, "rotowire_id", "rotowire_name")

                        play = Player(name, str(id), teamNamesArray[batterCounter])
                        print("Creating player object for player: " + name)

                        play.setPos(pos)
                        batterPlayerData.append(play)

                    if batterCounter == 0:
                        awayTeam.addBatters(batterPlayerData)
                    elif batterCounter == 1:
                        homeTeam.addBatters(batterPlayerData)

                    batterCounter += 1

                if batterCounter == 2:
                    print("Batters found for match, adding to lineup")

                    game.addTeam(awayTeam)
                    game.addTeam(homeTeam)
                    games.append(game)

            print("Lineups retrieved for team " + str(homeTeamName) + " and " + str(awayTeamName))

    return games

def readConstantsCSV(year):
    folder = 'data'
    name = 'MLBSeasonConstants.csv'
    with open('./' + folder + '/' + name, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == year:
                constants = wOBAFIPConstants(row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11],
                                 row[12, row[13]])
                return constants

def formTeams(baseballTable):
    tableData = baseballTable.find_all("tbody")[0]
    teamsHTML = tableData.find_all("th")
    teams = []

    for teamHTML in teamsHTML[:-1]:
        team = BaseballTeam(teamHTML.a['title'], teamHTML.a.text)
        teams.append(team)

    return teams

def getPitchingStats(pitchingRow, year, team):
    pStats = []
    for stat in pitchingStats:
        teamStat = pitchingRow.select_one('td[data-stat="' + stat + '"]').text
        if teamStat == '':
            teamStat = 0

        pStats.append(teamStat)

    pitchingTeamStats = PitchingTeam(int(pStats[0]), float(pStats[1]), float(pStats[2]), int(pStats[3]), int(pStats[4]),
                                     float(pStats[5]), float(pStats[6]), int(pStats[7]), int(pStats[8]), int(pStats[9]),
                                     int(pStats[10]), int(pStats[11]), int(pStats[12]), int(pStats[13]),
                                     float(pStats[14]), int(pStats[15]), int(pStats[16]), int(pStats[17]),
                                     int(pStats[18]),
                                     int(pStats[19]), int(pStats[20]), int(pStats[21]), int(pStats[22]),
                                     int(pStats[23]), int(pStats[24]),
                                     int(pStats[25]), int(pStats[26]), float(pStats[27]), float(pStats[28]),
                                     float(pStats[29]), float(pStats[30]),
                                     float(pStats[31]), float(pStats[32]), float(pStats[33]))

    return pitchingTeamStats

def getBattingStats(battingRow, year, team):
    stats = []
    for stat in battingStats:
        teamStat = battingRow.select_one('td[data-stat="' + stat + '"]').text
        stats.append(teamStat)

    battingTeamStats = BattingTeam(int(stats[0]), float(stats[1]), float(stats[2]), int(stats[3]), int(stats[4]),
                                   int(stats[5]),
                                   int(stats[6]), int(stats[7]), int(stats[8]), int(stats[9]), int(stats[10]),
                                   int(stats[11]), int(stats[12]),
                                   int(stats[13]), int(stats[14]), int(stats[15]), float(stats[16]), float(stats[17]),
                                   float(stats[18]), float(stats[19]),
                                   int(stats[20]), int(stats[21]), int(stats[22]), int(stats[23]), int(stats[24]),
                                   int(stats[25]), int(stats[26]), int(stats[27]))

    if year != 0 and team != '':
        folder = 'data' + '/' + str(year)
        name = 'TeamData.csv'
        with open('./' + folder + '/' + name, 'r') as f:
            reader = csv.reader(f)
            count = 0
            for row in reader:
                if row[0].find(team) and count > 0:
                    bSR = float(row[16])
                    battingTeamStats.addBSR(bSR)
                count += 1

    return battingTeamStats

def scrapeBaseballRefTeamData(yearsArrayIN):
    yearsArray = yearsArrayIN

    seasons = []
    for year in yearsArray:
        url = 'https://www.baseball-reference.com/leagues/MLB/' + str(year) + '.shtml'
        page = requests.get(url)

        soup = BeautifulSoup(page.text, 'html.parser')

        tables = soup.find_all("table")
        battersTable = tables[0]
        pitchersTable = ''

        for item in soup.find_all(text=lambda text: isinstance(text, Comment)):
            data = BeautifulSoup(item, "html.parser")
            for tab in data.find_all("table"):
                if len(tab.find_all('caption')):
                    if tab.find_all("caption")[0].text == 'Team Standard Pitching Table':
                        pitchersTable = tab

        teams = formTeams(battersTable)

        for team in teams:
            batterTableTeamRow = battersTable.select('a[title="' + team.team + '"]')[0].find_parent('tr')
            battingTeamStats = getBattingStats(batterTableTeamRow, year, team.team)
            team.addBatting(battingTeamStats)

            pitchingTeamRow = pitchersTable.select('a[title="' + team.team + '"]')[0].find_parent('tr')
            pitchingTeamStats = getPitchingStats(pitchingTeamRow, year, team.team)
            team.addPitching(pitchingTeamStats)


        season = Season(year, teams)

        batterFooterRow = battersTable.find('tfoot').find('tr')
        seasonBattingStats = getBattingStats(batterFooterRow, 0, '')
        season.battingTotals(seasonBattingStats)

        pitcherFooterRow = pitchersTable.find('tfoot').find('tr')
        seasonPitchingStats = getPitchingStats(pitcherFooterRow, 0, '')
        season.pitchingTotals(seasonPitchingStats)

        constants = readConstantsCSV(year)
        season.addwOBAFIP(constants)

        print(season)
        seasons.append(season)

    return seasons

def calcTeamPerformance(seasons):
    for season in seasons:
        for team in season.teams:
            wOBA = (((0.69 * team.batting.totBBs) + (0.72 * team.batting.totHBP)
                     + (0.89 * (team.batting.totHits - team.batting.totHRs - team.batting.totTriples - team.batting.totDoubles))
                     + (1.27 * team.batting.totDoubles) + (1.62 * team.batting.totTriples) + (2.10 * team.batting.totHRs))
                    /(team.batting.totAtBats + team.batting.totBBs - team.batting.totIBB + team.batting.totSF + team.batting.totHBP))
            hitsPerRun = team.batting.totHits / team.batting.totRuns
            iso = team.batting.avgSLG - team.batting.avgBAs
            kRate = team.batting.totSOs / team.batting.totPlateAppearances
            bbRate = team.batting.totBBs / team.batting.totPlateAppearances
            babip = ((team.batting.totHits - team.batting.totHRs)
                     / (team.batting.totAtBats - team.batting.totSOs - team.batting.totHRs + team.batting.totSF))
            team.batting.addWOBA(wOBA)
            team.batting.addHitsPerRun(hitsPerRun)
            team.batting.addISO(iso)
            team.batting.addKKRate(kRate)
            team.batting.addBBRate(bbRate)
            team.batting.addBABIP(babip)

    return seasons

def runRegressionAnalysis(train_x, train_y, target_x):
    train_x = np.array(train_x)
    train_y = np.array(train_y)

    alphas = [0.01, 0.001, 0.0001, 0.00001, 0.05, 0.005, 0.0005]
    reg = linear_model.RidgeCV(alphas=alphas, cv=4)

    reg.fit(train_x, train_y)
    print("alphas: %s" % alphas)
    print("Best alpha using built-in RidgeCV: %f" % reg.alpha_)


    alpha = reg.alpha_
    reg = linear_model.Ridge(alpha=alpha)
    reg.fit(train_x, train_y)
    target_y = reg.predict(target_x)

    return target_y

def regressTeamBattingPerformanceVsRuns(seasons):
    train_x = []
    train_y = []
    target_x = []
    target_y = []

    for season in seasons[:-1]:
        if season.season == MASTER_SEASONS[-2]:
            for team in season.teams:
                data = [team.batting.iso, team.batting.kRate, team.batting.bbRate, team.batting.BSR, team.batting.wOBA, team.batting.babip]
                target_x.append(data)
        else:
            for team in season.teams:
                data = [team.batting.iso, team.batting.kRate, team.batting.bbRate, team.batting.BSR, team.batting.wOBA, team.batting.babip]
                target = team.batting.totRuns
                train_x.append(data)
                train_y.append(target)

    target_y = runRegressionAnalysis(train_x, train_y, target_x)

    print("Team                     " + " Runs Scored " + " Projected Runs Scored " + " Batting Cluster Luck")
    for season in seasons[:-1]:
        if season.season == MASTER_SEASONS[-2]:
            counter = 0
            for team in season.teams:
                print(team.team + "     " + str(team.batting.totRuns) + "     " + str(round(target_y[counter]))
                      + "     " + str(round(((team.batting.totRuns) - (target_y[counter])))))
                team.batting.setClusterLuck(((team.batting.totRuns) - (target_y[counter])))
                team.batting.setProjectedRuns(target_y[counter])
                counter += 1

def regressTeamBattingPerformanceVsRunsCurrent(seasons):
    train_x = []
    train_y = []
    target_x = []
    target_y = []

    for season in seasons:
        if season.season == MASTER_SEASONS[-1]:
            for team in season.teams:
                data = [team.batting.iso, team.batting.kRate, team.batting.bbRate, team.batting.BSR, team.batting.wOBA, team.batting.babip]
                target_x.append(data)
        else:
            for team in season.teams:
                data = [team.batting.iso, team.batting.kRate, team.batting.bbRate, team.batting.BSR, team.batting.wOBA, team.batting.babip]
                target = team.batting.totRuns
                train_x.append(data)
                train_y.append(target)

    target_y = runRegressionAnalysis(train_x, train_y, target_x)

    print("Team                     " + " Runs Scored * Proportion " + " Projected Runs Scored " + " Batting Cluster Luck")
    for season in seasons:
        if season.season == MASTER_SEASONS[-1]:
            counter = 0
            for team in season.teams:
                team.batting.totRuns *= (GAMES_IN_SEASON / team.batting.games)

                print(team.team + "     " + str(team.batting.totRuns) + "     " + str(target_y[counter])
                      + "     " + str(((team.batting.totRuns) - (target_y[counter]))))
                team.batting.setClusterLuck(((team.batting.totRuns) - (target_y[counter])))
                team.batting.setProjectedRuns(target_y[counter])
                counter += 1

def regressTeamPitchingPerformanceVsRuns(seasons):
    train_x = []
    train_y = []
    target_x = []
    target_y = []

    for season in seasons[:-1]:
        if season.season == MASTER_SEASONS[-2]:
            for team in season.teams:
                data = [team.pitching.eraPlus, team.pitching.fip, team.pitching.whip, team.pitching.hr9, team.pitching.bb9, team.pitching.so9]
                target_x.append(data)
        else:
            for team in season.teams:
                data = [team.pitching.eraPlus, team.pitching.fip, team.pitching.whip, team.pitching.hr9, team.pitching.bb9, team.pitching.so9]
                target = team.pitching.runs
                train_x.append(data)
                train_y.append(target)

    target_y = runRegressionAnalysis(train_x, train_y, target_x)

    print("Team                     " + " Runs Allowed " + " Projected Runs Allowed " + " Pitching Cluster Luck")
    for season in seasons[:-1]:
        if season.season == MASTER_SEASONS[-2]:
            counter = 0
            for team in season.teams:
                print(team.team + "     " + str(team.pitching.runs) + "     " + str(target_y[counter])
                      + "     " + str(((team.pitching.runs) - (target_y[counter]))))
                team.pitching.setClusterLuck(((team.pitching.runs) - (target_y[counter])))
                team.pitching.setProjectedRuns(target_y[counter])
                counter += 1

def regressTeamPitchingPerformanceVsRunsCurrent(seasons):
    train_x = []
    train_y = []
    target_x = []
    target_y = []

    for season in seasons:
        if season.season == MASTER_SEASONS[-1]:
            for team in season.teams:
                data = [team.pitching.eraPlus, team.pitching.fip, team.pitching.whip, team.pitching.hr9, team.pitching.bb9, team.pitching.so9]
                target_x.append(data)
        else:
            for team in season.teams:
                data = [team.pitching.eraPlus, team.pitching.fip, team.pitching.whip, team.pitching.hr9, team.pitching.bb9, team.pitching.so9]
                target = team.pitching.runs
                train_x.append(data)
                train_y.append(target)

    target_y = runRegressionAnalysis(train_x, train_y, target_x)

    print("Team                     " + " Runs Allowed " + " Projected Runs Allowed " + " Pitching Cluster Luck")
    for season in seasons:
        if season.season == MASTER_SEASONS[-1]:
            counter = 0
            for team in season.teams:
                team.pitching.runs *= (GAMES_IN_SEASON / team.pitching.games)

                print(team.team + "     " + str(team.pitching.runs) + "     " + str(target_y[counter])
                      + "     " + str(((team.pitching.runs) - (target_y[counter]))))
                team.pitching.setClusterLuck(((team.pitching.runs) - (target_y[counter])))
                team.pitching.setProjectedRuns(target_y[counter])
                counter += 1

def getFangraphsDepthChartsROSProjectionsHelper(playerRows, arrayStats):
    statsPlayers = {}

    for playerRow in playerRows:
        stats = playerRow.find_all("td")
        playerName = ""
        statsDict = {}
        for i in range(len(arrayStats)):
            if i == 0:
                playerName = stats[i].a.text
                statsDict[arrayStats[i]] = stats[i].a.text
            elif i == 1:
                statsDict[arrayStats[i]] = ""
            else:
                statsDict[arrayStats[i]] = stats[i].text

        statsPlayers[playerName] = statsDict

    return statsPlayers

def getBaseballIDMappings():
    return pd.read_csv('http://crunchtimebaseball.com/master.csv', encoding='ISO-8859-1')

def getSecondaryBaseballIDMappings():
    return pd.read_csv('/Development/besbol/data/BaseballIDMappings.csv', encoding='ISO-8859-1')

def getFangraphsBatterProjections():
    return pd.read_csv('data/2019/DepthChartsBatterProjections2019.csv')

def getFangraphsPitcherProjections():
    return pd.read_csv('data/2019/DepthChartsPitcherProjections2019.csv')

def getFangraphsSeasonData(seasons):
    url_base = "https://www.fangraphs.com/projections.aspx?pos=all&stats={}&type=rfangraphsdc&team={}&lg=all&players=0"

    mappings = getBaseballIDMappings()

    for season in seasons:
        #id is 'ID' key
        batting_data = batting_stats(season.season)
        starter_data = pitching_stats(season.season, None, 'all', 1, 1, 'sta')
        reliever_data = pitching_stats(season.season, None, 'all', 1, 1, 'rel')

        # iterate through teams for each season
        # get players related to each team
        # for batting data, starter data, reliever data, create player objects
        # should get projections to create player objects in one fall swoop
        # add to batting and pitching (starters, relievers) for each team

        if season.season == MASTER_SEASONS[-1]:
            for i, row in batting_data.iterrows():
                if row['Team'] == "- - -":
                    try:
                        team = mappings.loc[row['fg_id'] == str(int(row['ID']))].iloc[0]['mlb_team']
                        batting_data.at[i, 'Team'] = abbreviationToFangraphsDict[team]
                        print("set team for: " + str(row['Name']))
                    except:
                        try:
                            team = mappings.loc[mappings['mlb_name'] == str(row['Name'])].iloc[0]['mlb_team']
                            batting_data.at[i, 'Team'] = abbreviationToFangraphsDict[team]
                            print("set team for: " + str(row['Name']))
                        except:
                            print("Could not set team for: " + str(row['Name']))

            for i, row in starter_data.iterrows():
                if row['Team'] == "- - -":
                    try:
                        team = mappings.loc[row['fg_id'] == str(int(row['ID']))].iloc[0]['mlb_team']
                        starter_data.at[i, 'Team'] = abbreviationToFangraphsDict[team]
                        print("set team for: " + str(row['Name']))
                    except:
                        try:
                            team = mappings.loc[mappings['mlb_name'] == str(row['Name'])].iloc[0]['mlb_team']
                            starter_data.at[i, 'Team'] = abbreviationToFangraphsDict[team]
                            print("set team for: " + str(row['Name']))
                        except:
                            print("Could not set team for: " + str(row['Name']))

            for i, row in reliever_data.iterrows():
                if row['Team'] == "- - -":
                    try:
                        team = mappings.loc[row['fg_id'] == str(int(row['ID']))].iloc[0]['mlb_team']
                        reliever_data.at[i, 'Team'] = abbreviationToFangraphsDict[team]
                        print("set team for: " + str(row['Name']))
                    except:
                        try:
                            team = mappings.loc[mappings['mlb_name'] == str(row['Name'])].iloc[0]['mlb_team']
                            reliever_data.at[i, 'Team'] = abbreviationToFangraphsDict[team]
                            print("set team for: " + str(row['Name']))
                        except:
                            print("Could not set team for: " + str(row['Name']))

        for team in season.teams:
            fangraphsTeam = bbrefToFangraphsDict[team.team]
            team.fgTeamName = fangraphsTeam

            team_batting_data = batting_data.loc[batting_data['Team'] == fangraphsTeam]
            team_starter_data = starter_data.loc[starter_data['Team'] == fangraphsTeam]
            team_reliever_data = reliever_data.loc[reliever_data['Team'] == fangraphsTeam]

            team_batting_data_dict = team_batting_data.to_dict('records')
            team_starter_data_dict = team_starter_data.to_dict('records')
            team_reliever_data_dict = team_reliever_data.to_dict('records')

            batters = []
            for batter in team_batting_data_dict:
                id = -1
                try:
                    id = mappings.loc[mappings['fg_id'] == str(int(batter['ID']))].iloc[0]['mlb_id']
                except:
                    try:
                        id = mappings.loc[mappings['mlb_name'] == str(batter['Name'])].iloc[0]['mlb_id']
                    except:
                        print("id not found for: " + str(batter['Name']))

                player = Player(batter['Name'], str(id), team.team)
                player.setSeasonStats(batter)
                batters.append(player)

            starters = []
            for starter in team_starter_data_dict:
                id = -1
                try:
                    id = mappings.loc[mappings['fg_id'] == str(int(starter['ID']))].iloc[0]['mlb_id']
                except:
                    try:
                        id = mappings.loc[mappings['mlb_name'] == str(starter['Name'])].iloc[0]['mlb_id']
                    except:
                        print("id not found for: " + str(starter['Name']))

                player = Player(starter['Name'], str(id), team.team)
                player.setSeasonStats(starter)
                starters.append(player)

            relievers = []
            for reliever in team_reliever_data_dict:
                id = -1
                try:
                    id = mappings.loc[mappings['fg_id'] == str(int(reliever['ID']))].iloc[0]['mlb_id']
                except:
                    try:
                        id = mappings.loc[mappings['mlb_name'] == str(reliever['Name'])].iloc[0]['mlb_id']
                    except:
                        print("id not found for: " + str(reliever['Name']))

                player = Player(reliever['Name'], str(id), team.team)
                player.setSeasonStats(reliever)
                relievers.append(player)

            if season.season == MASTER_SEASONS[-1]:
                print("Getting projections for: " + str(team.team))
                for pos in ["bat", "pit"]:
                    team_id = fangraphsToTeamDict[team.team]

                    url = url_base.format(pos, team_id)
                    page = requests.get(url)
                    soup = BeautifulSoup(page.text, 'html.parser')

                    tableData = soup.find_all("table")[3]
                    data = tableData.find_all("tbody")[0]

                    playerRows = data.find_all("tr")

                    if pos == "bat":
                        statsBatDict = getFangraphsDepthChartsROSProjectionsHelper(playerRows, fangraphsProjBattingStats)
                        for player in batters:
                            if player.playerName in statsBatDict:
                                player.setProjections(statsBatDict[player.playerName])

                        team.batting.addBatters(batters)

                    if pos == "pit":
                        statsPitDict = getFangraphsDepthChartsROSProjectionsHelper(playerRows, fangraphsProjPitchingStats)
                        for player in relievers:
                            if player.playerName in statsPitDict:
                                player.setProjections(statsPitDict[player.playerName])

                        for player in starters:
                            if player.playerName in statsPitDict:
                                player.setProjections(statsPitDict[player.playerName])

                        team.pitching.addPitchers(starters)
                        team.pitching.addRelievers(relievers)

                    time.sleep(1)
            else:
                team.batting.addBatters(batters)
                team.pitching.addPitchers(starters)
                team.pitching.addRelievers(relievers)

    return seasons

def compareAndProjSeasons(previousSeason, currentSeason):
    totRunsScored = 0
    totRunsAllowed = 0

    for team in currentSeason.teams:
        print("Current WAR Calcs for : " + team.team)

        battersTotalWar = 0
        for batter in team.batting.batters:
            currentWar = 0
            projectedWar = 0
            if bool(batter.seasonStats):
                currentWar = batter.seasonStats['WAR']
            if bool(batter.projection):
                projectedWar = batter.projection['WAR']

            batterTotWar = float(currentWar) + float(projectedWar)
            battersTotalWar += batterTotWar

        print("Batters WAR: " + str(battersTotalWar))

        team.batting.batterWar = battersTotalWar

        startersTotalWar = 0
        for starter in team.pitching.pitchers:
            currentWar = 0
            projectedWar = 0

            if bool(starter.seasonStats):
                currentWar = starter.seasonStats['WAR']
            if bool(starter.projection):
                projectedWar = starter.projection['WAR']

            starterTotWar = float(currentWar) + float(projectedWar)
            startersTotalWar += starterTotWar

        print("Starters WAR: " + str(startersTotalWar))

        team.pitching.pitcherWar = startersTotalWar

        relieversTotalWar = 0
        for reliever in team.pitching.relievers:
            currentWar = 0
            projectedWar = 0

            if bool(reliever.seasonStats):
                currentWar = reliever.seasonStats['WAR']
            if bool(reliever.projection):
                projectedWar = reliever.projection['WAR']

            reliefTotalWar = float(currentWar) + float(projectedWar)
            relieversTotalWar += reliefTotalWar

        print("Relievers WAR: " + str(relieversTotalWar))

        team.pitching.relieverWar = relieversTotalWar

    for team in previousSeason.teams:
        print("Previous WAR Calcs for : " + team.team)

        battersTotalWar = 0
        for batter in team.batting.batters:
            if bool(batter.seasonStats):
                batterTotWar = batter.seasonStats['WAR']
                battersTotalWar += float(batterTotWar)

        print("Batters WAR: " + str(battersTotalWar))

        team.batting.batterWar = battersTotalWar

        startersTotalWar = 0
        for starter in team.pitching.pitchers:
            if bool(starter.seasonStats):
                starterTotWar = starter.seasonStats['WAR']
                startersTotalWar += float(starterTotWar)

        print("Starters WAR: " + str(startersTotalWar))

        team.pitching.pitcherWar = startersTotalWar

        relieversTotalWar = 0
        for reliever in team.pitching.relievers:
            if bool(reliever.seasonStats):
                reliefTotalWar = reliever.seasonStats['WAR']
                relieversTotalWar += float(reliefTotalWar)

        print("Relievers WAR: " + str(relieversTotalWar))

        team.pitching.relieverWar = relieversTotalWar

    for currentTeam in currentSeason.teams:
        print("Projection for: " + currentTeam.team)

        prevTeam = currentTeam
        for dumbTeam in previousSeason.teams:
            if currentTeam.abbreviation == dumbTeam.abbreviation:
                prevTeam = dumbTeam
                break

        currentTeam.batting.changeInBattingWarFromPrevYear = currentTeam.batting.batterWar - prevTeam.batting.batterWar
        currentTeam.batting.changeInBattingRunsFromPrevYear = currentTeam.batting.changeInBattingWarFromPrevYear * R_W
        print("Change in Batting Runs: " + str(currentTeam.batting.changeInBattingRunsFromPrevYear))

        currentTeam.pitching.changeInPitchingWarFromPrevYear = prevTeam.pitching.pitcherWar - currentTeam.pitching.pitcherWar
        currentTeam.pitching.changeInPitchingRunsFromPrevYear = currentTeam.pitching.changeInPitchingWarFromPrevYear * R_W
        print("Change in Starting Pitching Runs: " + str(currentTeam.pitching.changeInPitchingRunsFromPrevYear))

        currentTeam.pitching.changeInReliefWarFromPrevYear = prevTeam.pitching.relieverWar - currentTeam.pitching.relieverWar
        currentTeam.pitching.changeInReliefRunsFromPrevYear = currentTeam.pitching.changeInReliefWarFromPrevYear * R_W
        print("Change in Relief Runs: " + str(currentTeam.pitching.changeInReliefRunsFromPrevYear))

        currentTeam.batting.projectedRSYear = round(int(prevTeam.batting.totRuns) +
                                                prevTeam.batting.luck + currentTeam.batting.changeInBattingRunsFromPrevYear)
        totRunsScored += currentTeam.batting.projectedRSYear

        currentTeam.pitching.projectedRAYear = round(int(prevTeam.pitching.runs) +
                                                 prevTeam.pitching.luck + currentTeam.pitching.changeInReliefRunsFromPrevYear +
                                                 currentTeam.pitching.changeInPitchingRunsFromPrevYear)
        totRunsAllowed += currentTeam.pitching.projectedRAYear

    runNormalization = (totRunsAllowed + totRunsScored) / 2

    for currentTeam in currentSeason.teams:
        print("Projection for: " + currentTeam.team)
        currentTeam.batting.projectedRSPercentYear = currentTeam.batting.projectedRSYear / totRunsScored
        currentTeam.pitching.projectedRAPercentYear = currentTeam.pitching.projectedRAYear / totRunsAllowed

        currentTeam.batting.projectedRSYear = currentTeam.batting.projectedRSPercentYear * runNormalization
        print("Projected Runs Scored: " + str(currentTeam.batting.projectedRSYear))

        currentTeam.pitching.projectedRAYear = currentTeam.pitching.projectedRAPercentYear * runNormalization
        print("Projected Runs Allowed: " + str(currentTeam.pitching.projectedRAYear))

        currentTeam.projWinPercentageYear = ((pow(currentTeam.batting.projectedRSYear, 1.83))
                                         / (pow(currentTeam.batting.projectedRSYear, 1.83)
                                            + pow(currentTeam.pitching.projectedRAYear, 1.83)))
        print("Projected Winning %: " + str(currentTeam.projWinPercentageYear))

        currentTeam.projWinsYear = round(GAMES_IN_SEASON * currentTeam.projWinPercentageYear)
        currentTeam.projLosesYear = round(GAMES_IN_SEASON * (1 - currentTeam.projWinPercentageYear))

        print("Projected W-L")
        print(str(currentTeam.projWinsYear) + " - " + str(currentTeam.projLosesYear))

def adjustedWARHelper(startBatter, projectionsForBatters):
    currWar = 0
    projWar = 0
    currGames = 0
    projGames = GAMES_IN_SEASON

    if projectionsForBatters is not None:
        print("Projections for: " + str(startBatter.playerName))

        if 'WAR' in projectionsForBatters.seasonStats:
            currWar = float(projectionsForBatters.seasonStats['WAR'])
        else:
            print('Season WAR not found for: ' + str(projectionsForBatters.playerName))

        if 'G' in projectionsForBatters.seasonStats:
            currGames = float(projectionsForBatters.seasonStats['G'])
        else:
            print('Season G not found for: ' + str(projectionsForBatters.playerName))

        if 'WAR' in projectionsForBatters.projection:
            projWar = float(projectionsForBatters.projection['WAR'])
        else:
            print('Projection WAR not found for: ' + str(projectionsForBatters.playerName))

        if 'G' in projectionsForBatters.projection:
            projGames = float(projectionsForBatters.projection['G'])
        else:
            print('Projection G not found for: ' + str(projectionsForBatters.playerName))

        print("Current War: " + str(currWar))
        print("Current Games: " + str(currGames))
        print("Projected War: " + str(projWar))
        print("Projected Games: " + str(projGames))
    else:
        print("Projection/Season Stats for batter not found: " + str(startBatter.playerName))

    adjWar = float((currWar + projWar)) * (float(GAMES_IN_SEASON) / float((currGames + projGames)))

    print("Adjusted War: " + str(adjWar))
    startBatter.setDailyAdjWar(adjWar)

    return startBatter

def retrievePreseasonProjections(batterProjections, pitcherProjections, startBatter, batter):
    try:
        if batter:
            print("Preseason projection for: " + str(startBatter.playerName))
            WAR = batterProjections.loc[batterProjections['Name'] == str(startBatter.playerName)].iloc[0]['WAR']
            G = batterProjections.loc[batterProjections['Name'] == str(startBatter.playerName)].iloc[0]['G']
            print("WAR: " + str(WAR))
            print("G: " + str(G))
            projections = {'WAR': WAR, 'G': G}
            startBatter.setProjections(projections)

            print("Retrieved batter preseason projections for: " + str(startBatter.playerName))

            return startBatter
        else:
            print("Preseason projection for: " + str(startBatter.playerName))
            WAR = pitcherProjections.loc[pitcherProjections['Name'] == str(startBatter.playerName)].iloc[0]['WAR']
            G = pitcherProjections.loc[pitcherProjections['Name'] == str(startBatter.playerName)].iloc[0]['G']
            print("WAR: " + str(WAR))
            print("G: " + str(G))
            projections = {'WAR': WAR, 'G': G}
            startBatter.setProjections(projections)

            print("Retrieved pitcher batter preseason projections for: " + str(startBatter.playerName))

            return startBatter
    except:
        print("Did not find preseason projections for player: " + str(startBatter.playerName))

def calculateWARAdjustments(season, matchups):
    teams = season.teams

    batterProjections = getFangraphsBatterProjections()
    pitcherProjections = getFangraphsPitcherProjections()

    # Low hanging fruit
    # TODO: adjust for home field advantage
    # TODO: make sure to adjust for "the opener" strategy
    # TODO: park factors/weather playing into team's strength
    # TODO: players playing different positions
    # TODO: Calculate Exponent for win %
    # TODO: Pickle file to save data

    # Medium
    # TODO: take into account pinch hitters
    # TODO: Relievers
    # TODO: splits
    # TODO: best algo to decide projections (is it regression?)

    # High
    # TODO: monte carlo simulation
    # TODO: backtesting w/ ROS fangraphs projections

    for match in matchups:
        for currTeam in match.teams:
            # find team object
            seasonTeam = teams[0]
            foundTeam = False
            print("Team: " + str(currTeam.teamName))
            for team in teams:
                if currTeam.teamName == team.fgTeamName:
                    seasonTeam = team
                    foundTeam = True
                    break

            if not foundTeam:
                print("did not find team: " + str(currTeam.teamName))
            # iterate through matchup batters -> find batter object in team
            # change matchup batter projectedWar via games played/projected -> season stats

            print("_____________")
            print("Batters: ")
            count = 0
            for startBatter in currTeam.batters:
                projectionsForBatters = None
                foundBatter = False
                for bat in seasonTeam.batting.batters:
                    if bat.id == startBatter.id:
                        projectionsForBatters = bat
                        foundBatter = True
                        break

                if not foundBatter:
                    projectionsForBatters = retrievePreseasonProjections(batterProjections, pitcherProjections, startBatter, True)

                currTeam.batters[count] = adjustedWARHelper(startBatter, projectionsForBatters)

                count += 1

            projectionsForPitchers = None
            foundPitcher = False

            print("_____________")
            print("Pitcher: ")

            for pit in seasonTeam.pitching.pitchers:
                if pit.id == currTeam.startingPitcher.id:
                    foundPitcher = True
                    projectionsForPitchers = pit
                    break

            if not foundPitcher:
                for pit in seasonTeam.pitching.relievers:
                    if pit.id == currTeam.startingPitcher.id:
                        foundPitcher = True
                        projectionsForPitchers = pit
                        break

                if not foundPitcher:
                    projectionsForPitchers = retrievePreseasonProjections(batterProjections, pitcherProjections,
                                                                          currTeam.startingPitcher, False)

                currTeam.startingPitcher = adjustedWARHelper(currTeam.startingPitcher, projectionsForPitchers)

            else:
                currTeam.startingPitcher = adjustedWARHelper(currTeam.startingPitcher, projectionsForPitchers)

            if currTeam.opener is not None:
                print("retrieving opener and starter")
                projectionsForOpener = None
                foundOpener = False
                for pit in seasonTeam.pitching.pitchers:
                    if pit.id == currTeam.opener.id:
                        foundOpener = True
                        projectionsForOpener = pit
                        break

                if not foundOpener:
                    for pit in seasonTeam.pitching.relievers:
                        if pit.id == currTeam.opener.id:
                            foundOpener = True
                            projectionsForOpener = pit
                            break

                    if not foundOpener:
                        projectionsForOpener = retrievePreseasonProjections(batterProjections, pitcherProjections,
                                                                              currTeam.opener, False)

                    currTeam.opener = adjustedWARHelper(currTeam.opener, projectionsForOpener)

                else:
                    currTeam.opener = adjustedWARHelper(currTeam.opener, projectionsForOpener)

            currTeam.adjustedPitchingWar = 0

            if currTeam.opener is not None:
                currTeam.adjustedPitchingWar = currTeam.startingPitcher.dailyAdjustedWAR * (8/9)\
                                               + currTeam.opener.dailyAdjustedWAR * (1/9)
                print("Calculated adjusted war for pitching opener")
            else:
                currTeam.adjustedPitchingWar = currTeam.startingPitcher.dailyAdjustedWAR

            for startBatter in currTeam.batters:
                currTeam.adjustedBatterWar += startBatter.dailyAdjustedWAR

            currTeam.adjustedRelieverWar = seasonTeam.pitching.relieverWar

            currTeam.changeBatterWar = currTeam.adjustedBatterWar - seasonTeam.batting.batterWar
            currTeam.changePitchingWar = currTeam.adjustedPitchingWar + currTeam.adjustedRelieverWar\
                                         - seasonTeam.pitching.pitcherWar - seasonTeam.pitching.relieverWar

            currTeam.projectedRS = round(int(seasonTeam.batting.totRuns) + seasonTeam.batting.luck +
                                         (currTeam.changeBatterWar * R_W))
            currTeam.projectedRA = round(int(seasonTeam.pitching.runs) - seasonTeam.pitching.luck -
                                         ((currTeam.changePitchingWar) * R_W))

            match.totRunsScored += currTeam.projectedRS

            match.totRunsAllowed += currTeam.projectedRA

        match.runNormalization = (match.totRunsAllowed + match.totRunsScored) / 2

    for match in matchups:
        print("---------------------------------")
        for currTeam in match.teams:
            print("Projection for: " + currTeam.teamName)
            currTeam.projectedRSPercent = currTeam.projectedRS / match.totRunsScored
            currTeam.projectedRAPercent = currTeam.projectedRA / match.totRunsAllowed

            currTeam.projectedRS = currTeam.projectedRSPercent * match.runNormalization
            print("Projected Adjusted Runs Scored: " + str(currTeam.projectedRS))

            currTeam.projectedRA = currTeam.projectedRAPercent * match.runNormalization
            print("Projected Adjusted Runs Allowed: " + str(currTeam.projectedRA))

            currTeam.winningPercentage = ((pow(currTeam.projectedRS, 1.83))
                                                 / (pow(currTeam.projectedRS, 1.83)
                                                    + pow(currTeam.projectedRA, 1.83)))
            print("Projected Winning %: " + str(currTeam.winningPercentage))

if __name__ == "__main__":
    year = 2019
    month = 6
    day = 19
    filename = "data/" + str(year) + "/daily/season/" + str(month) + "_" + str(day) + "_" + "data.p"
    filenameLineup = "data/" + str(year) + "/daily/lineup/" + str(month) + "_" + str(day) + "_" + "data.p"

    seasons = None

    if (path.exists(filename)):
        seasons = pickle.load(open(filename, "rb"))
    else:
        seasons = scrapeBaseballRefTeamData(MASTER_SEASONS)
        seasons = calcTeamPerformance(seasons)
        seasons = getFangraphsSeasonData(seasons)

        regressTeamBattingPerformanceVsRuns(seasons)
        regressTeamPitchingPerformanceVsRuns(seasons)

        pickle.dump(seasons, open(filename, "wb"))

    # compare 2018 players with 2019 players
    compareAndProjSeasons(seasons[-2], seasons[-1])

    regressTeamBattingPerformanceVsRunsCurrent(seasons)
    regressTeamPitchingPerformanceVsRunsCurrent(seasons)

    if (path.exists(filenameLineup)):
        matchups = pickle.load(open(filenameLineup, "rb"))
    else:
        matchups = get_lineups()
        pickle.dump(matchups, open(filenameLineup, "wb")) # TOOD: get datafsfsc

    # get current projected war and then difference with the starting lineup =

    calculateWARAdjustments(seasons[-1], matchups)

    '''
    Algorithm Logic

    Season has future projections

    Daily Games:

    Classifer for projecting runs from team data
        - Pitching
        - Batting
        - feature scale runs based off of how many times they have played

    Take into account cluster luck for runs actually scored/allowed

    Add modifications based off of war for players in lineup and their projections
        - Batting - add starting lineup up assume war
            - compare against projected runs
        - Pitching
            - Starters: games / projected starts = multiplier for war (assume playing every day)
                - Compare against runs allowed
            - Relievers add war/projected up all together

    Apply home field advantage

    Calculate Exponent -> then calculate win percentage

    Perform machine learning on same pitcher/batter features -> classifer is from regression over players over past five years
        - Reliever classifer
        - Starter classifer
        - Batter classifer

    Project using features what should be expected runs allowed/runs scored
    Identify cluster luck per player -> agregate over pitchers
    '''

    # Angels - Felix Pena
    # Athetics - Daniel Mengden
    '''
    Brewers
    Marlins
    Rays
    Tigers
    Braves
    Pirates
    '''
