from gensim.models import KeyedVectors
from gensim.models.fasttext import FastText
import konlpy

if __name__=='__main__':
	pretrained_fasttext = KeyedVectors.load_word2vec_format('data/pretrained_embedding_vectors/cc.ko.300.vec', encoding='utf-8', unicode_errors='ignore')

	# # Getting the tokens
	# words = []
	# for word in pretrained_fasttext.vocab:
	# 	words.append(word)

	### using konlpy library to split sentences to words
	processor = konlpy.tag.Kkma()
	input_path = 'data/processed/train_embedding.txt'
	i = 0
	with open(input_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as file_input:
		sentences = file_input.readlines()
		# for sentence in file_input:
		# 	i += 1
		# 	new_words = processor.nouns(sentence)
		# 	words.extend(new_words)

	pretrained_fasttext.save('source/models/fasttext_gensim.model')
	pretrained_fasttext = FastText.load('source/models/fasttext_gensim.model')
	# fasttext_model = FastText(size=300, window=50, min_count=7, sample=1e-2, sg=1, iter=100)
	fasttext_model=pretrained_fasttext
	fasttext_model.build_vocab(sentences)
	fasttext_model.train(sentences, total_examples=fasttext_model.corpus_count, epochs=fasttext_model.iter)
	fasttext_model.wv.save("source/models/fasttext_gensim.model")



	# Pick a word
	keyword = '애니메이션 시리즈'
	keyword = '김우빈'
	keyword = '설리'
	keyword = '구하라'
	keyword = '경제'
	keyword = '티엔'
	keyword = '유투브'
	keyword = '경제'
	# # Finding out similar words [default= top 10]
	# pretrained_fasttext.distance('베트남', '한국')
	# pretrained_fasttext.distance('베트남', '미국')
	# pretrained_fasttext.distance('미국', '한국')
	# pretrained_fasttext.distance('사이곤', '호치민')
	# pretrained_fasttext.distance('햄버거', '한국')
	# pretrained_fasttext.distance('베트남', '햄버거')
	# pretrained_fasttext.distance('일본', '한국')
	# print('------------------ {} --------------------'.format(keyword))
	# for similar_word in pretrained_fasttext.similar_by_word(keyword):
	# 	print("Word: {0}, -------- : {1:.2f}".format(similar_word[0], similar_word[1]))
