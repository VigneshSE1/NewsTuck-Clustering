import fasttext
#model = fasttext.train_unsupervised('D:/Tamil_News_Clustering/myTrainData.txt', "cbow")
# model = fasttext.train_unsupervised("D:/Tamil_News_Clustering/PROJECT_DATA/TRAINING_DATA/formated_data/non_com/Adaderana_.txt")
# model = fasttext.train_unsupervised("D:/Tamil_News_Clustering/PROJECT_DATA/TRAINING_DATA/formated_data/non_com/bbc_.txt")
# model = fasttext.train_unsupervised("D:/Tamil_News_Clustering/PROJECT_DATA/TRAINING_DATA/formated_data/non_com/data_.txt")
# model = fasttext.train_unsupervised("D:/Tamil_News_Clustering/PROJECT_DATA/TRAINING_DATA/formated_data/non_com/ITN_.txt")
# model = fasttext.train_unsupervised("D:/Tamil_News_Clustering/PROJECT_DATA/TRAINING_DATA/formated_data/non_com/NewsFirst_.txt")
# model = fasttext.train_unsupervised("D:/Tamil_News_Clustering/PROJECT_DATA/TRAINING_DATA/formated_data/non_com/Sooriyan_FM_News_.txt")
# model = fasttext.train_unsupervised("D:/Tamil_News_Clustering/PROJECT_DATA/TRAINING_DATA/formated_data/non_com/Thinakaran_.txt")
# model = fasttext.train_unsupervised("D:/Tamil_News_Clustering/PROJECT_DATA/TRAINING_DATA/formated_data/non_com/Thinakkural_.txt")
#model = fasttext.train_unsupervised("D:/Tamil_News_Clustering/PROJECT_DATA/TRAINING_DATA/formated_data/non_com/Virakesari_.txt")

#model.save_model("clus.bin")
model = fasttext.load_model("clus.bin")
print(model.get_nearest_neighbors("கைது"))