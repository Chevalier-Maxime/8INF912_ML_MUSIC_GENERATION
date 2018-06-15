from bs4 import BeautifulSoup
import urllib3
import urllib.request
import os
from argparse import ArgumentParser

def remove(value):
    deletechars = r'\/:*?"<>|' #Unotorized Windows characters
    for c in deletechars:
        value = value.replace(c,'')
    value = value.replace('\n','')
    return value

def crawler(url, game):

    http = urllib3.PoolManager()
    response = http.request('GET', url)
    soup = BeautifulSoup(response.data,"html5lib")

    cwd = os.getcwd()
    if not os.path.exists(cwd + os.sep + "Musics"):
        os.makedirs(cwd + os.sep + "Musics")

    cwd = cwd + os.sep + "Musics"

    game_name = None

    row_index = 1

    rows = soup.find_all("tr")

    #print("nb lignes : {}".format(len(rows)))
    while row_index < len(rows):
        row = rows[row_index]
        #On itere sur les lignes
        #print(row,"row class : ", row.get("class"))
        if(row.get("class") == ['header']):
            if game == None or game in row.getText():
                # On a le nom du jeu
                game_name = remove(row.getText())
                print(game_name)
                #Construction du dossier
                game_path = (cwd + os.sep + game_name).rstrip()
                print(game_path)
                if not os.path.exists(game_path):
                    os.makedirs(game_path)
                #Tant qu'on arrive pas sur un autre jeu
                row_index+=1
                row = rows[row_index]
                while (row.get("class") != ['header']):
                    #Pour chaque colone <--- plus simple possible mais si on a besoin de
                    #l'info d'un des lignes on peut facilement la reccuperer
                    for col in row.find_all("td"):
                        link = col.find('a', href=True)
                        if((link != None) and (".mid" in link.get("href"))):
                            #Telecharger le lien
                            song_name = remove(link.get("href"))
                            url_game = url+song_name
                            print(song_name)
                            urllib.request.urlretrieve(url_game, game_path+os.sep+song_name)
                    row_index+=1
                    if(row_index < len(rows)):
                        row = rows[row_index]
                    else:
                        break
            else:
                if(row_index < len(rows)):
                    row_index+=1
                else:
                    break
        else:
            row_index+=1
    



#for header in soup.find_all("tr", {"class": "header"}):
#    for link in header.find_all("a"):
#        print(link)
#        if(".mid" in link.getText()):
#            #On a le nom du fichier
#            print(link)
def get_argument_parser():
    parser = ArgumentParser()

    # Mandatory arguments
    #parser.add_argument('output_folder', type=str, help='The folder where \
    #        the .mid files will be stored')
    parser.add_argument('--game',
                         default='nes',
                         const='nes',
                         nargs='?',
                         choices=['nes','ff'],
                         help='The type of data to crawl. \
                                Default: %(default)s')

    return parser

if __name__ == "__main__":
    args = (get_argument_parser()).parse_args()

    print(args)
    if args.game == 'nes':
        url = 'https://www.vgmusic.com/music/console/nintendo/nes/'
        crawler(url, None)
    elif args.game == 'ff':
        url = 'https://www.vgmusic.com/music/console/nintendo/nes/'
        crawler(url, 'Final Fantasy')
        url = 'https://www.vgmusic.com/music/console/nintendo/snes/'
        crawler(url, 'Final Fantasy')
        url = 'https://www.vgmusic.com/music/console/sony/ps1/'
        crawler(url, 'Final Fantasy')
    else:
        print("Should not append")

