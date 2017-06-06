instructions for running MUSE - Modularizing Unsupervised Sense Embeddings

Note:

	(1) tensorflow version: 0.11.0rc1.
	
	(2) python2.7.
	
	(3) You can either conduct pre-processing by yourself following step 1 and step 2, or you can safely download the pre-processed dataset from (http://speech.ee.ntu.edu.tw/~homer/preprocessing.zip), unzip it, and replace the folder preprocessing/ with the unzipped one.

	(4) To test the MUSE model, you can either conduct experiments by yourself following step 1 to step 4, or you can download the pre-trained models from (http://speech.ee.ntu.edu.tw/~homer/one-sided_optimization.zip and http://speech.ee.ntu.edu.tw/~homer/original_formulation.zip and http://speech.ee.ntu.edu.tw/~homer/value_function_factorization.zip).
	WARNING: each zip file is about 10GB.

	(5) Please feel free to contact me if you have any problem. Thanks.

1. Download requirements:

	(1) Download Wikipedia 2010 dump preprocessed by Westbury lab (http://www.psych.ualberta.ca/~westburylab/downloads/westburylab.wikicorp.download.html), and put it into folder "preprocessing/".
	
	(2) Download Stanford NLP tools.
	
	(3) Download Multi-Sense Skip-gram (MSSG) from (http://iesl.cs.umass.edu/downloads/vectors/release.tar.gz), and put "vectors.MSSG.300D.6K" into folder "preprocessing/".
	
	(4) Download Stanford's Contextual Word Similarities (SCWS) dataset: http://www-nlp.stanford.edu/~ehhuang/SCWS.zip. Please put the "ratings.txt" into folder "preprocessing".
	
	(5) Synonym selection datasets are not public, please ask the original authors for the datasets. Thanks.

2. Preprocess training corpus:

	(0) "cd preprocessing".
	
	(1) Set "stanford-parser.jar" and "stanford-postagger.jar" in Stanford NLP tools into your CLASSPATH.
	
	(2) Run "python parse.py" to get tokenized corpus "TokenizedCorpus.txt".
	
	(3) Run "python makeSenseN.py" to get word ID file "wordID_MSSG.txt".	(currently in Odin:prepare_data/release)
	
	(4) Run "python make_train_sensplit.py TokenizedCorpus.txt train.sensplit.txt" to get sentence tokenized corpus "train.sensplit.txt" as well as unigram word ID file "wordID_MSSG_unigram_sensplit.txt".
	
	(5)	Run "python make_test_sensplit.py ratings.txt test.sensplit.txt" to get sentence tokenized testing SCWS dataset "test.sensplit.txt".
	
	(6)	Great! now you have all necessary files (train.sensplit.txt, test.sensplit.txt, wordID_MSSG_unigram_sensplit.txt) for model training.

3. Perform model training by running "MUSE_train.py".

	(1)	Before running the program, please create a directory for checkpoint storage. Lots of space is needed to store the checkpoint in every epoch and the best model parameters for MaxSimC and AvgSimC.
	
	(2) Run "python MUSE_train.py -h" to see available parameter settings. Three parameters must be specified: --dataset_dir for the directory containing [train.sensplit.txt, test.sensplit.txt, wordID_MSSG_unigram_sensplit.txt], --save_dir for the directory for checkpoint storage, and --log_path for logging the model performance. The valid range of the parameters are listed as follows:

		--formulation {original_formulation,one-sided_optimization,value_function_factorization}
		--learning_method {Q-greedy/Q-epsilon-greedy/Q-Boltzmann/policy/smooth_policy}
		
		--lr_rate double
		--memory double (between 0 and 1, GPU memory fraction)
		--ckpt_threshold double (best between 0 and 0.6)

		--batch_size integer
		--sample_size integer
		--context_window integer
		--sense_number integer
		--selection_dim integer
		--representation_dim integer

		--dataset_dir string  (directory)
		--save_dir string (directory)
		--log_path string

	(3) Perform model training. For example, using
		"python MUSE_train.py --memory 0.1 --dataset_dir /home/homer/preprocessing/ --log_path oneside_greedy.log --save_dir /tmp2/final_models/oneside_greedy/ --learning_method Q-greedy --formulation one-sided_optimization" 
	to build MUSE model using one-sided optimization and Q-greedy learning strategy.
	Some examples are also shown on the bottom of this page.

4. Perform model testing by running "MUSE_test.py".

	(1) Run "python MUSE_test.py -h" to see available parameter settings. Two parameters must be specified: --ckpt_path for the path of tested checkpoint file, and --dataset_dir for the directory containing [train.sensplit.txt, test.sensplit.txt, wordID_MSSG_unigram_sensplit.txt]. For the rest of parameters, please use default parameters if you didn't alter the parameters during training. Otherwise, you should adjust the parameters to its corresponding value during training. 
	
	(2) Perform model testing. For example, "python MUSE_test.py --context_window 5 --ckpt_path /tmp2/final_models/oneside_greedy/lr_0.025_window_5_batch_2048_sample_25_sense_3-bestMaxC-15000 --dataset_dir /home/homer/preprocessing/". Once the code is run, the MaxSimC and AvgSimC values will appear. Afterwards, there will be 5 testing options illustrated in the program and as below. Please note that all instructions are SEPARATED BY TAB.

	(3) The first testing option is the test the kNN in each sense for a specified word. The instruction template is "1'\t'word", where "word" is the tested word.
	For example, using "1	head" shows the following result.
		
		k-NN sorted by collocation likelihood
		sense 1
		venter thorax neck spear millimeters fusiform beachy shaved maldives whale
		sense 2
		shaved thatcher loki thorax mao luthor chest pressure kryptonite neck
		sense 3
		multi-party appoints unicameral beria appointed minister-president cabinet thatcher elect coach
	
	(4) The second testing option is to test the probabilistic policy as well as kNN for sense selection given a text context. The instruction template is "2'\t'sentence'\t'word_index", where word_index is the index starting from 0 specifying the position of the target word in the sentence for sense selection.
	For example, using "2	appoint john pope republican as head of the new army of	5" shows the following result.
		
		probability for each sense:
		[ 0.21802232  0.21920459  0.56277311]
		k-NN sorted by collocation likelihood
		sense 1
		venter thorax neck spear millimeters fusiform beachy shaved maldives whale
		sense 2
		shaved thatcher loki thorax mao luthor chest pressure kryptonite neck
		sense 3
		multi-party appoints unicameral beria appointed minister-president cabinet thatcher elect coach
	
	(5) The third testing option is to test the contexts in the training corpus, where each sense of the specified word will be selected. The instruction template is "3'\t'word'\t'probability_threshold", where the probability_threshold is the threshold that skips a sentence for the target word without the maximum sense selection probability passing the threshold.
	For example, using "3	head	0.33" shows the following result.
		
		k-NN sorted by collocation likelihood
		sense 1
		venter thorax neck spear millimeters fusiform beachy head shaved maldives
		sense 2
		shaved thatcher loki thorax mao luthor chest pressure kryptonite neck
		sense 3
		multi-party appoints unicameral beria appointed minister-president cabinet thatcher elect coach

		.
		.
		.
		.
		.
		.
		.
		.
		.
		.
		.
		.
		.
		.
		.
		example for sense 1
		cones punchbowl diamond head koko head includes bay koko crater salt
		shells and/or high explosive squash head hesh and/or anti-tank guided missiles
		while affluent ladies wore extravagant head ornaments combs pearl necklaces face
		cast george anderson justus barnes head bandit walter cameron sheriff
		duck reservation clerk daisy duck head waiter goofy mascot pluto mechanical
		==================
		example for sense 2
		is handicapped slower car getting head start using an index lowest
		albums paul boutique check your head ill communication and hello nasty
		previous bbc news programming includes head head your news and bbc
		bbc news programming includes head head your news and bbc news
		head was shaven to prevent head lice serious threat back then
		==================
		example for sense 3
		traced to of an ox head in egyptian hieroglyph or the
		catch up to tortoise with head start and therefore that motion
		appoint john pope republican as head of the new army of
		appointed republican ambrose burnside to head the army of the potomac
		single-shot caliber mm at his head firing at point-blank range
		==================
	
	(6) The fourth testing option is to test the accuracy of synonym selection using the TOEFL dataset. Please specified the TOEFL dataset folder in the instruction.
	For example, using "4	toefl" shows the following result.

		('...', 'testing toefl')
		Accuracy = 0.811594202899
		... testing toefl done in 0:00:00.081102

	(7) The fifth testing option is to test the accuracy of synonym selection using the ESL-50/RD-300 dataset. Please specified the dataset in the instruction. Each line in the dataset should be formatted as:
	"question_word | answer_word | synonym_candidate_2 | synonym_candidate_3 | synonym_candidate_4"
	For example, using "5	esl-rd/RD300.txt" shows the following result.

		('...', 'testing esl-rd/RD300.txt')
		Accuracy = 0.589285714286
		... testing esl-rd/RD300.txt done in 0:00:00.054413

	(8) The sixth testing option is to dump the sense embeddings and corresponding indices. The instruction format should be: "6\tdump_embedding_file\tdump_index_file" 
	For example, "6\tembed.txt\tindex.tsv".0

We also provide some example training settings here:

	(1) python MUSE_train.py --memory 0.1 --dataset_dir /home/homer/preprocessing/ --log_path original_smooth_policy.log --save_dir /tmp2/final_models/original_smooth_policy/ --learning_method smooth_policy --formulation original_formulation

	(2) python MUSE_train.py --memory 0.1 --dataset_dir /home/homer/preprocessing/ --log_path oneside_smooth_policy.log --save_dir /tmp2/final_models/oneside_smooth_policy/ --learning_method smooth_policy --formulation one-sided_optimization

	(3) python MUSE_train.py --memory 0.1 --dataset_dir /home/homer/preprocessing/ --log_path original_policy.log --save_dir /tmp2/final_models/original_policy/ --learning_method policy --formulation original_formulation

	(4) python MUSE_train.py --memory 0.1 --dataset_dir /home/homer/preprocessing/ --log_path oneside_policy.log --save_dir /tmp2/final_models/oneside_policy/ --learning_method policy --formulation one-sided_optimization

	(5) python MUSE_train.py --memory 0.1 --dataset_dir /home/homer/preprocessing/ --log_path original_greedy.log --save_dir /tmp2/final_models/original_greedy/ --learning_method Q-greedy --formulation original_formulation

	(6) python MUSE_train.py --memory 0.1 --dataset_dir /home/homer/preprocessing/ --log_path oneside_greedy.log --save_dir /tmp2/final_models/oneside_greedy/ --learning_method Q-greedy --formulation one-sided_optimization

	(7) python MUSE_train.py --memory 0.15 --dataset_dir /home/homer/preprocessing/ --log_path factorization_greedy.log --save_dir /tmp2/final_models/factorization_greedy/ --learning_method Q-greedy --formulation value_function_factorization

	(8) python MUSE_train.py --memory 0.1 --dataset_dir /home/homer/preprocessing/ --log_path original_epsilon_greedy.log --save_dir /tmp2/final_models/original_epsilon_greedy/ --learning_method Q-epsilon-greedy --formulation original_formulation

	(9) python MUSE_train.py --memory 0.1 --dataset_dir /home/homer/preprocessing/ --log_path oneside_epsilon_greedy.log --save_dir /tmp2/final_models/oneside_epsilon_greedy/ --learning_method Q-epsilon-greedy --formulation one-sided_optimization

	(10) python MUSE_train.py --memory 0.15 --dataset_dir /home/homer/preprocessing/ --log_path factorization_epsilon_greedy.log --save_dir /tmp2/final_models/factorization_epsilon_greedy/ --learning_method Q-epsilon-greedy --formulation value_function_factorization

	(11) python MUSE_train.py --memory 0.1 --dataset_dir /home/homer/preprocessing/ --log_path original_boltzmann.log --save_dir /tmp2/final_models/original_boltzmann/ --learning_method Q-Boltzmann --formulation original_formulation

	(12) python MUSE_train.py --memory 0.1 --dataset_dir /home/homer/preprocessing/ --log_path oneside_boltzmann.log --save_dir /tmp2/final_models/oneside_boltzmann/ --learning_method Q-Boltzmann --formulation one-sided_optimization

	(13) python MUSE_train.py --memory 0.15 --dataset_dir /home/homer/preprocessing/ --log_path factorization_boltzmann.log --save_dir /tmp2/final_models/factorization_boltzmann/ --learning_method Q-Boltzmann --formulation value_function_factorization
