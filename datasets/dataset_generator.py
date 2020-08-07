import os
import time

from pypdb import describe_pdb, get_all_info, describe_pdb
from tqdm import tqdm, tqdm_gui

ec_path = "/content/drive/My Drive/Colab Notebooks/Old_DMCNN/datasets/EC.txt"
checked_path = "/content/drive/My Drive/Colab Notebooks/Old_DMCNN/datasets/checked.txt"
dict_path = ""

ec = open(ec_path, "w")
checked = open(checked_path, "w")

checked_list = read_list(checked_path)

class list_generator(object):

	def __init__(self, split=0.1, max_size=1200, min_size=400, save=True, check=False):
		max_size = self.max_size
		min_size = self.min_size

		#Force check of PDB if its been a week since last update
		if self.check_time() > one week
			self.check = True
			print("Weekly force check of PDB - may take a minute")
		
		if self.check == True:
			self.load_data()	
			self.check_pdb()
			self.split()
			if self.save == True
				self.save_lists()
		else:
			self.load_lists()

	def check_time(self):
		last_time = open("last_time.txt",'r')
		last_time = int(last_time.read())
		
		return time.time() - last_time

	def check_pdb(self):
		pdb_ids = pypdb.get_all()
		checked_list = #LOAD CHECKED list

		for pdb_id in checked_list:
			if pdb_id in pdb_ids:
				pdb_ids.remove(pdb_id)

		for pdb_id in pdb_ids:
			self.fetch_info(pdb_id)

		#UPDATE CHECK TIME TOO

	def fetch_info(self, pdb_ids):
		for pdb_id in tqdm(pdb_ids, disc="Checking PDB for new structures", unit="Structures "):
			try:
				info = get_all_info(pdb_id)
				enzyClass = info["polymer"]["enzClass"]["@ec"]
				length = int(info["polymer"]["@length"])
				#ADD TO DATA FILE
			except:
				pass
			add_to_checked(pdb_id)

	def load_data(self):
		ec_dict = csv.load(dict_path)
		#PROCESS HERE
		self.ec_dict = ec_dict

	def split(self):
		for i in pdb_ids:

		self.training =
		self.validation =

	def save_lists(self):
		training_name = 'training_{min_size}_{max_size}.txt'.format(max_size=max_size, min_size=min_size)
		validation_name = 'validation_{min_size}_{max_size}.txt'.format(max_size=max_size, min_size=min_size)
		#Delete current lists
		os.remove(training_name)
		os.remove(validation_name)
		#Save new lists
		with open(training_name, 'w') as f:
			for pdb_id in self.training:
				f.write("%s, " % pdb_id)

		with open(validation_name, 'w') as f:
			for pdb_id in self.validation:
				f.write("%s, " % pdb_id)

	def load_lists(self):
		file_name = '{min_size}_{max_size}.txt'.format(max_size=max_size, min_size=min_size)
		self.whole_list = read_list(file_name)
		self.split()






for i in tqdm(checked_list):
	if i in pdb_ids:
		pdb_ids.remove(i)

for pdb_id in tqdm(pdb_ids):
	pdb_info = describe_pdb(pdb_id)
	AAs = int(pdb_info["nr_residues"])

	if AAs <= 400:
		if AAs > 200:
			info = get_all_info(pdb_id)
			try:
				enzyClass = info["polymer"]["enzClass"]["@ec"]
				ec.write("{}, ".format(pdb_id))
			except:
				pass
		else:
			pass
	else:
		pass
	checked.write("{}, ".format(pdb_id))


def read_list(path):
	"Reads list stored in txt file"
	list_enzymes = open(path,'r')
	list_enzymes = list_enzymes.read()
	return list(list_enzymes.split(", "))