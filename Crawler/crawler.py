from bs4 import BeautifulSoup
import urllib3
import urllib.request
import os

http = urllib3.PoolManager()

url = 'https://www.vgmusic.com/music/console/nintendo/nes/'
response = http.request('GET', url)
soup = BeautifulSoup(response.data,"lxml")

cwd = os.getcwd()
if not os.path.exists(cwd + os.sep + "Musics"):
    os.makedirs(cwd + os.sep + "Musics")

cwd = cwd + os.sep + "Musics"

game_name = None

row_index = 1

rows = soup.find_all("tr")

print("nb lignes : {}".format(len(rows)))
while row_index < len(rows):
    row = rows[row_index]
    #On itere sur les lignes
    #print(row,"row class : ", row.get("class"))
    if(row.get("class") == ['header']):
        # On a le nom du jeu
        game_name = row.getText().replace('\n','')
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
                    song_name = link.get("href")
                    url_game = url+song_name
                    print(song_name)
                    urllib.request.urlretrieve(url_game, game_path+os.sep+song_name)
            row_index+=1
            row = rows[row_index]
    else:
        row_index+=1
    



#for header in soup.find_all("tr", {"class": "header"}):
#    for link in header.find_all("a"):
#        print(link)
#        if(".mid" in link.getText()):
#            #On a le nom du fichier
#            print(link)


