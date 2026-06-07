from worldcup_predictor.data import download_kaggle, kaggle_metadata
print(kaggle_metadata().get("titleNullable"))
print(download_kaggle())
