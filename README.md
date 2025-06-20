# Validatie van automatische modellen voor segmentatie van de lichaamssamenstelling op L3
In deze repositroy zijn de python scripts te vinden die gebruikt zijn voor het extraheren van de segmentatiemodellen AutoMATiCA en CompositIA. 

/AutoMATiCA-master (1) bestaat uit slechts een Yaml file met de packages voor het extraheren van AutoMATiCA. De originele pythoncode, vrijgeven door M Paris, is niet aangepast. AutoMATiCA is geactiveerd op een lokale Windows computer met gebruik van CPU (Microsoft Windows 10 Enterprise). Het is ook mogelijk om gebruik te maken van GPU indien u de instructies volgt, beschreven in de GitHub van M. Paris.

/CompositIA-master (2) bestaat uit een aangepast python code van R. Cabini. De orginele code maakt voorgaand aan het automatisch segmenteren, automatische L3 selectie mogeljik. Dit gaat voorbij de scope van ons onderzoek. De env-compositia.yml file bevat alle packages om CompositIA te activeren op Linux systeem (Rocky Linux 8.10 (Green Obsidian)). In de compositia.slurm bevat de slurm sbatch job om in het cluster van het LUMC de scans te segmenteren. weigths_unet_l3.txt bevat een link naar de hd5f file met alle unets weights voor het l3 segmentatie model.

/anlysis bevat alle pythonscripts die het mogelijk maken om de verkregen segmentations uit AutoMATiCA en CompositIA te vergelijken met de groundtruth tag files van het LUMC. De gebruikte packages zijn te vinden in env-analysis.yml .

/statistics bevat alle pythonscripts voor het toepassen en het weergeven van de statistische toetsen. De gebruikte packages zijn te vindein in env-analysis.yml .






1. Paris MT, Mourtzakis M. Automated body composition analysis of clinically acquired computed tomography scans using neural networks. Clin Nutr. 2020;39(10):3049–3055.
2. Cabini RF, Cozzi A, Coppola F, Morganti S, et al. CompositIA: an open-source automated quantification tool for body composition scores from thoraco-abdominal CT scans. Eur Radiol Exp. 2025;9:12. https://doi.org/10.1186/s41747-025-00552-7
