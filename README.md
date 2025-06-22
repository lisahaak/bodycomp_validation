# Validatie van automatische modellen voor segmentatie van de lichaamssamenstelling op L3
#### KTO-AS24029 Lichaamssamenstelling-LUMC 24020

In deze repository zijn de Python-scripts te vinden die gebruikt zijn voor het extraheren en vergelijken van de segmentatiemodellen **AutoMATiCA** en **CompositIA**.

## Inhoud van de repository

### `/AutoMATiCA-master` [¹]
Bevat enkel een `yaml`-bestand met de vereiste packages voor het activeren van AutoMATiCA.  
De originele Python-code, vrijgegeven door M. Paris, is niet aangepast. [GitHub-pagina van M. Paris](https://github.com/MicheleParis/AutoMATiCA).

- Uitgevoerd op een lokale Windows-computer met CPU (Microsoft Windows 10 Enterprise).
- Gebruik van een GPU is mogelijk door de instructies te volgen op de 

### `/CompositIA-master` [²]
Bevat een aangepaste versie van de oorspronkelijke Python-code van R. Cabini. [GitHub-pagina van R. Cabini] (https://github.com/rcabini/compositIA).


- De originele code bevat automatische L3-selectie, maar dit valt buiten de scope van ons onderzoek.
- Het bestand `env-compositia.yml` bevat alle benodigde packages om CompositIA te activeren op  Rocky Linux 8.10 (Green Obsidian).
- `compositia.slurm` bevat een SLURM sbatch script voor het uitvoeren van segmentatie op het LUMC-cluster.
- `weights_unet_l3.txt` bevat een link naar het `.hdf5`-bestand met de U-Net weights voor het L3-segmentatiemodel.

### `/analysis`
Bevat Python-scripts om de segmentaties van AutoMATiCA en CompositIA te vergelijken met de groundtruth tag-bestanden van het LUMC.

- Benodigde packages zijn gespecificeerd in `env-analysis.yml`.

### `/statistics`
Bevat Python-scripts voor het uitvoeren en visualiseren van statistische toetsen.

- Gebruikte packages zijn eveneens opgenomen in `env-analysis.yml`.

## Referenties

1. Paris MT, Mourtzakis M. Automated body composition analysis of clinically acquired computed tomography scans using neural networks. *Clinical Nutrition*. 2020;39(10):3049–3055. doi: 10.1016/j.clnu.2020.01.008.
2. Cabini RF, Cozzi A, Coppola F, Morganti S, et al. CompositIA: an open-source automated quantification tool for body composition scores from thoraco-abdominal CT scans. *European Radiology Experimental*. 2025;9:12. doi: 10.1186/s41747-025-00552-7.


