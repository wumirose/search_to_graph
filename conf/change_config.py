from config import Config

def change_config(a, b, x, y, z):
    Config.searchquery = a
    Config.retmax = b
    Config.WC = x
    Config.DF = y
    Config.JSONDOC = z

def addtional_config_chamge(p):
    Config.PARSED = p