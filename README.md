# Validatie van automatische modellen voor segmentatie van de lichaamssamenstelling op L3
In deze repositroy zijn de python scripts te vinden die gebruikt zijn voor het extraheren van de segmentatiemodellen AutoMATiCA en CompositIA. 

/AutoMATiCA-master bestaat uit slechts een Yaml file met de packages voor het extraheren van AutoMATiCA. De originele pythoncode, vrijgeven door M Paris, is niet aangepast. AutoMATiCA is geactiveerd op een lokale Windows computer met gebruik van CPU (Microsoft Windows 10 Enterprise). Het is ook mogelijk om gebruik te maken van GPU indien u de instructies volgt, beschreven in de GitHub van M. Paris.

/CompositIA-master bestaat uit een aangepast python code van R. Cabini. De orginele code maakt voorgaand aan het automatisch segmenteren, automatische L3 selectie mogeljik. Dit gaat voorbij de scope van ons onderzoek. De env-compositia.yml file bevat alle packages om CompositIA te activeren op Linux systeem (Rocky Linux 8.10 (Green Obsidian)). 

/anlysis bevat alle pythonscripts die het mogelijk maken om de verkregen segmentations uit AutoMATiCA en CompositIA te vergelijken met de groundtruth tag files van het LUMC. De gebruikte packages zijn te vinden in env-analysis.yml .

/statistics bevat alle pythonscripts voor het toepassen en het weergeven van de statistische toetsen. De gebruikte packages zijn te vindein in env-analysis.yml .
