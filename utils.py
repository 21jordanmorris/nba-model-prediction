from basketball_reference_scraper.teams import get_roster, get_team_stats, get_opp_stats, get_roster_stats, get_team_misc
import pandas as pd

def getTeamMisc(year):
    atl = get_team_misc('ATL', year).to_frame().transpose()
    brk = get_team_misc('BRK', year).to_frame().transpose()
    bos = get_team_misc('BOS', year).to_frame().transpose()
    cho = get_team_misc('CHO', year).to_frame().transpose()
    chi = get_team_misc('CHI', year).to_frame().transpose()
    cle = get_team_misc('CLE', year).to_frame().transpose()
    dal = get_team_misc('DAL', year).to_frame().transpose()
    den = get_team_misc('DEN', year).to_frame().transpose()
    det = get_team_misc('DET', year).to_frame().transpose()
    gsw = get_team_misc('GSW', year).to_frame().transpose()
    hou = get_team_misc('HOU', year).to_frame().transpose()
    ind = get_team_misc('IND', year).to_frame().transpose()

    lac = get_team_misc('LAC', year).to_frame().transpose()
    lal = get_team_misc('LAL', year).to_frame().transpose()
    mem = get_team_misc('MEM', year).to_frame().transpose()
    mia = get_team_misc('MIA', year).to_frame().transpose()
    mil = get_team_misc('MIL', year).to_frame().transpose()
    mint = get_team_misc('MIN', year).to_frame().transpose()
    nop = get_team_misc('NOP', year).to_frame().transpose()
    nyk = get_team_misc('NYK', year).to_frame().transpose()
    okc = get_team_misc('OKC', year).to_frame().transpose()
    orl = get_team_misc('ORL', year).to_frame().transpose()
    phi = get_team_misc('PHI', year).to_frame().transpose()
    pho = get_team_misc('PHO', year).to_frame().transpose()

    por = get_team_misc('POR', year).to_frame().transpose()
    sac = get_team_misc('SAC', year).to_frame().transpose()
    sas = get_team_misc('SAS', year).to_frame().transpose()
    tor = get_team_misc('TOR', year).to_frame().transpose()
    uta = get_team_misc('UTA', year).to_frame().transpose()
    was = get_team_misc('WAS', year).to_frame().transpose()

    frames = [atl,brk,bos,cho,chi,cle,dal,den,det,gsw,hou,ind,lac,lal,mem,mia,mil,mint,nop,nyk,okc,orl,phi,pho,por,sac,sas,tor,uta,was]
    return pd.concat(frames)