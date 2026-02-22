#!/bin/bash
# Submit tokenization jobs for all medical configs
# Adjust --nodes per line as needed
# ex: TOKENIZATION_MODE=paired sbatch --nodes=20 --job-name=tok-llavaOvmid /users/rkreft/benchmark-image-tokenzier/vision_tokenization/scripts/run_multinode_tokenization.slurm --config /users/rkreft/benchmark-image-tokenzier/vision_tokenization/configs/llava_onevision_midtrain_paired_to_emu3.5_apertus_image2text.json --offline --resume

## MEDICAL ##
#TOKENIZATION_MODE=image_only sbatch --nodes=2 --job-name=tok-busi /users/rkreft/benchmark-image-tokenzier/vision_tokenization/scripts/run_multinode_tokenization.slurm --config /users/rkreft/benchmark-image-tokenzier/vision_tokenization/configs/apertus_s1_medical/medical-busi-imgonly-emu3p5.json --offline
#TOKENIZATION_MODE=image_only sbatch --nodes=2 --job-name=tok-covid_us /users/rkreft/benchmark-image-tokenzier/vision_tokenization/scripts/run_multinode_tokenization.slurm --config /users/rkreft/benchmark-image-tokenzier/vision_tokenization/configs/apertus_s1_medical/medical-covid_us-imgonly-emu3p5.json --offline
#TOKENIZATION_MODE=image_only sbatch --nodes=2 --job-name=tok-ddti /users/rkreft/benchmark-image-tokenzier/vision_tokenization/scripts/run_multinode_tokenization.slurm --config /users/rkreft/benchmark-image-tokenzier/vision_tokenization/configs/apertus_s1_medical/medical-ddti-imgonly-emu3p5.json --offline
#TOKENIZATION_MODE=image_only sbatch --nodes=2 --job-name=tok-isic /users/rkreft/benchmark-image-tokenzier/vision_tokenization/scripts/run_multinode_tokenization.slurm --config /users/rkreft/benchmark-image-tokenzier/vision_tokenization/configs/apertus_s1_medical/medical-isic-imgonly-emu3p5.json --offline
#TOKENIZATION_MODE=image_only sbatch --nodes=10 --job-name=tok-mammoth /users/rkreft/benchmark-image-tokenzier/vision_tokenization/scripts/run_multinode_tokenization.slurm --config /users/rkreft/benchmark-image-tokenzier/vision_tokenization/configs/apertus_s1_medical/medical-mammoth-imgonly-emu3p5.json --offline
#TOKENIZATION_MODE=image_only sbatch --nodes=4 --job-name=tok-medmax /users/rkreft/benchmark-image-tokenzier/vision_tokenization/scripts/run_multinode_tokenization.slurm --config /users/rkreft/benchmark-image-tokenzier/vision_tokenization/configs/apertus_s1_medical/medical-medmax-imgonly-emu3p5.json --offline
#TOKENIZATION_MODE=image_only sbatch --nodes=2 --job-name=tok-medmnist /users/rkreft/benchmark-image-tokenzier/vision_tokenization/scripts/run_multinode_tokenization.slurm --config /users/rkreft/benchmark-image-tokenzier/vision_tokenization/configs/apertus_s1_medical/medical-medmnist-imgonly-emu3p5.json --offline
#TOKENIZATION_MODE=image_only sbatch --nodes=20 --job-name=tok-medtrinity_full /users/rkreft/benchmark-image-tokenzier/vision_tokenization/scripts/run_multinode_tokenization.slurm --config /users/rkreft/benchmark-image-tokenzier/vision_tokenization/configs/apertus_s1_medical/medical-medtrinity_full-imgonly-emu3p5.json --offline
#TOKENIZATION_MODE=image_only sbatch --nodes=40 --job-name=tok-open_pmc_18m /users/rkreft/benchmark-image-tokenzier/vision_tokenization/scripts/run_multinode_tokenization.slurm --config /users/rkreft/benchmark-image-tokenzier/vision_tokenization/configs/apertus_s1_medical/medical-open_pmc_18m-imgonly-emu3p5.json --offline
#TOKENIZATION_MODE=image_only sbatch --nodes=10 --job-name=tok-pmc_oa /users/rkreft/benchmark-image-tokenzier/vision_tokenization/scripts/run_multinode_tokenization.slurm --config /users/rkreft/benchmark-image-tokenzier/vision_tokenization/configs/apertus_s1_medical/medical-pmc_oa-imgonly-emu3p5.json --offline
#TOKENIZATION_MODE=image_only sbatch --nodes=1 --job-name=tok-rfmid2 /users/rkreft/benchmark-image-tokenzier/vision_tokenization/scripts/run_multinode_tokenization.slurm --config /users/rkreft/benchmark-image-tokenzier/vision_tokenization/configs/apertus_s1_medical/medical-rfmid2-imgonly-emu3p5.json --offline
#TOKENIZATION_MODE=image_only sbatch --nodes=1 --job-name=tok-scin /users/rkreft/benchmark-image-tokenzier/vision_tokenization/scripts/run_multinode_tokenization.slurm --config /users/rkreft/benchmark-image-tokenzier/vision_tokenization/configs/apertus_s1_medical/medical-scin-imgonly-emu3p5.json --offline
#TOKENIZATION_MODE=image_only sbatch --nodes=1 --job-name=tok-slide /users/rkreft/benchmark-image-tokenzier/vision_tokenization/scripts/run_multinode_tokenization.slurm --config /users/rkreft/benchmark-image-tokenzier/vision_tokenization/configs/apertus_s1_medical/medical-slide-imgonly-emu3p5.json --offline
#TOKENIZATION_MODE=image_only sbatch --nodes=1 --job-name=tok-uwf /users/rkreft/benchmark-image-tokenzier/vision_tokenization/scripts/run_multinode_tokenization.slurm --config /users/rkreft/benchmark-image-tokenzier/vision_tokenization/configs/apertus_s1_medical/medical-uwf-imgonly-emu3p5.json --offline

## HOLO ASSIST ##
#TOKENIZATION_MODE=image_only sbatch --nodes=10 --job-name=tok-holoassist /users/rkreft/benchmark-image-tokenzier/vision_tokenization/scripts/run_multinode_tokenization.slurm --config /users/rkreft/benchmark-image-tokenzier/vision_tokenization/configs/apertus_s1_egocentric/egocentric-holoassist-imgonly-emu3p5.json --offline

## General ##

# WDS - laion aesthetics12m-umap
TOKENIZATION_MODE=image_only sbatch --nodes=20 --job-name=tok-laionaesthetic /users/rkreft/benchmark-image-tokenzier/vision_tokenization/scripts/run_multinode_tokenization.slurm --data-format wds --config /users/rkreft/benchmark-image-tokenzier/vision_tokenization/configs/apertus_s1_general/general-laion_aesthetics12m_uwp-imgonly-emu3p5_wds.json
# WDS - Mint1T PDF
TOKENIZATION_MODE=image_only sbatch --nodes=15 --job-name=tok-mint1tpdf_01 /users/rkreft/benchmark-image-tokenzier/vision_tokenization/scripts/run_multinode_tokenization.slurm --data-format wds --config /users/rkreft/benchmark-image-tokenzier/vision_tokenization/configs/apertus_s1_general/general-mint1t_pdf_01-imgonly-emu3p5_wds.json
TOKENIZATION_MODE=image_only sbatch --nodes=15 --job-name=tok-mint1tpdf_02 /users/rkreft/benchmark-image-tokenzier/vision_tokenization/scripts/run_multinode_tokenization.slurm --data-format wds --config /users/rkreft/benchmark-image-tokenzier/vision_tokenization/configs/apertus_s1_general/general-mint1t_pdf_02-imgonly-emu3p5_wds.json
TOKENIZATION_MODE=image_only sbatch --nodes=15 --job-name=tok-mint1tpdf_03 /users/rkreft/benchmark-image-tokenzier/vision_tokenization/scripts/run_multinode_tokenization.slurm --data-format wds --config /users/rkreft/benchmark-image-tokenzier/vision_tokenization/configs/apertus_s1_general/general-mint1t_pdf_03-imgonly-emu3p5_wds.json
TOKENIZATION_MODE=image_only sbatch --nodes=15 --job-name=tok-mint1tpdf_04 /users/rkreft/benchmark-image-tokenzier/vision_tokenization/scripts/run_multinode_tokenization.slurm --data-format wds --config /users/rkreft/benchmark-image-tokenzier/vision_tokenization/configs/apertus_s1_general/general-mint1t_pdf_04-imgonly-emu3p5_wds.json
TOKENIZATION_MODE=image_only sbatch --nodes=8 --job-name=tok-mint1tpdf_05 /users/rkreft/benchmark-image-tokenzier/vision_tokenization/scripts/run_multinode_tokenization.slurm --data-format wds --config /users/rkreft/benchmark-image-tokenzier/vision_tokenization/configs/apertus_s1_general/general-mint1t_pdf_05-imgonly-emu3p5_wds.json --resume
# streaming mode hf webdataset integration for more shards(streaming = False because with streaming num shards is max num tar files)
TOKENIZATION_MODE=image_only sbatch --nodes=32 --job-name=tok-mint1tpdf_06 /users/rkreft/benchmark-image-tokenzier/vision_tokenization/scripts/run_multinode_tokenization.slurm --config /users/rkreft/benchmark-image-tokenzier/vision_tokenization/configs/apertus_s1_general/general-mint1t_pdf_06-imgonly-emu3p5_wds.json
TOKENIZATION_MODE=image_only sbatch --nodes=64 --job-name=tok-mint1tpdf_07 /users/rkreft/benchmark-image-tokenzier/vision_tokenization/scripts/run_multinode_tokenization.slurm --config /users/rkreft/benchmark-image-tokenzier/vision_tokenization/configs/apertus_s1_general/general-mint1t_pdf_07-imgonly-emu3p5_wds.json
# WDS - Mint1T HTML first 20 %
TOKENIZATION_MODE=image_only sbatch --nodes=200 --job-name=tok-mint1thtml /users/rkreft/benchmark-image-tokenzier/vision_tokenization/scripts/run_multinode_tokenization.slurm --data-format wds --config /users/rkreft/benchmark-image-tokenzier/vision_tokenization/configs/apertus_s1_general/general-mint1t_HTML-imgonly-emu3p5_wds.json --resume



## OCR ##

# MINT 1T Arxiv
TOKENIZATION_MODE=image_only sbatch --nodes=32 --job-name=tok-mint1tarxiv /users/rkreft/benchmark-image-tokenzier/vision_tokenization/scripts/run_multinode_tokenization.slurm --config /users/rkreft/benchmark-image-tokenzier/vision_tokenization/configs/apertus_s1_ocr/ocr-mint1t_arxiv-imgonly-emu3p5_wds.json
#Big Docs 7.5 M
TOKENIZATION_MODE=image_only sbatch --nodes=10 --job-name=tok-bigdocs7_5m /users/rkreft/benchmark-image-tokenzier/vision_tokenization/scripts/run_multinode_tokenization.slurm --data-format wds --config /users/rkreft/benchmark-image-tokenzier/vision_tokenization/configs/apertus_s1_ocr/ocr-bigdocs7_5m-imgonly-emu3p5_wds_arxiv_ocr.json
TOKENIZATION_MODE=image_only sbatch --nodes=2 --job-name=tok-bigdocs7_5m /users/rkreft/benchmark-image-tokenzier/vision_tokenization/scripts/run_multinode_tokenization.slurm --data-format wds --config /users/rkreft/benchmark-image-tokenzier/vision_tokenization/configs/apertus_s1_ocr/ocr-bigdocs7_5m-imgonly-emu3p5_wds_arxiv_tablecap.json
TOKENIZATION_MODE=image_only sbatch --nodes=3 --job-name=tok-bigdocs7_5m /users/rkreft/benchmark-image-tokenzier/vision_tokenization/scripts/run_multinode_tokenization.slurm --data-format wds --config /users/rkreft/benchmark-image-tokenzier/vision_tokenization/configs/apertus_s1_ocr/ocr-bigdocs7_5m-imgonly-emu3p5_wds_cocotext.json
TOKENIZATION_MODE=image_only sbatch --nodes=1 --job-name=tok-bigdocs7_5m /users/rkreft/benchmark-image-tokenzier/vision_tokenization/scripts/run_multinode_tokenization.slurm --data-format wds --config /users/rkreft/benchmark-image-tokenzier/vision_tokenization/configs/apertus_s1_ocr/ocr-bigdocs7_5m-imgonly-emu3p5_wds_cordv2.json
TOKENIZATION_MODE=image_only sbatch --nodes=10 --job-name=tok-bigdocs7_5m /users/rkreft/benchmark-image-tokenzier/vision_tokenization/scripts/run_multinode_tokenization.slurm --data-format wds --config /users/rkreft/benchmark-image-tokenzier/vision_tokenization/configs/apertus_s1_ocr/ocr-bigdocs7_5m-imgonly-emu3p5_wds_pubtables1m.json
TOKENIZATION_MODE=image_only sbatch --nodes=1 --job-name=tok-bigdocs7_5m /users/rkreft/benchmark-image-tokenzier/vision_tokenization/scripts/run_multinode_tokenization.slurm --data-format wds --config /users/rkreft/benchmark-image-tokenzier/vision_tokenization/configs/apertus_s1_ocr/ocr-bigdocs7_5m-imgonly-emu3p5_wds_textocr.json

# Handwritte nicola
TOKENIZATION_MODE=image_only sbatch --nodes=2 --job-name=tok-nicolaHwDat /users/rkreft/benchmark-image-tokenzier/vision_tokenization/scripts/run_multinode_tokenization.slurm --config /users/rkreft/benchmark-image-tokenzier/vision_tokenization/configs/apertus_s1_ocr/ocr-nicola-handwriting-imgonly-emu3p5_data.json
TOKENIZATION_MODE=image_only sbatch --nodes=1 --job-name=tok-nicolaHwSlide /users/rkreft/benchmark-image-tokenzier/vision_tokenization/scripts/run_multinode_tokenization.slurm --config /users/rkreft/benchmark-image-tokenzier/vision_tokenization/configs/apertus_s1_ocr/ocr-nicola-handwriting-imgonly-emu3p5_data_slides.json


## BIO ##
TOKENIZATION_MODE=image_only sbatch --nodes=11 --job-name=tok-treeoflife10m /users/rkreft/benchmark-image-tokenzier/vision_tokenization/scripts/run_multinode_tokenization.slurm --config /users/rkreft/benchmark-image-tokenzier/vision_tokenization/configs/apertus_s1_bio/bio-treeoflife10m-imgonly-emu3p5_wds.json

## Sattelite img + Geo ##
TOKENIZATION_MODE=image_only sbatch --nodes=16 --job-name=tok-swissimg /users/rkreft/benchmark-image-tokenzier/vision_tokenization/scripts/run_multinode_tokenization.slurm --data-format wds --config /users/rkreft/benchmark-image-tokenzier/vision_tokenization/configs/apertus_s1_map_geo/geo-swissimage-imgonly-emu3p5_wds.json
TOKENIZATION_MODE=image_only sbatch --nodes=1 --job-name=tok-bigearthnet /users/rkreft/benchmark-image-tokenzier/vision_tokenization/scripts/run_multinode_tokenization.slurm --data-format wds --config /users/rkreft/benchmark-image-tokenzier/vision_tokenization/configs/apertus_s1_map_geo/geo-bigearthnet-imgonly-emu3p5_wds.json
TOKENIZATION_MODE=image_only sbatch --nodes=1 --job-name=tok-dfc2020 /users/rkreft/benchmark-image-tokenzier/vision_tokenization/scripts/run_multinode_tokenization.slurm --data-format wds --config /users/rkreft/benchmark-image-tokenzier/vision_tokenization/configs/apertus_s1_map_geo/geo-dfc2020-imgonly-emu3p5_wds.json
