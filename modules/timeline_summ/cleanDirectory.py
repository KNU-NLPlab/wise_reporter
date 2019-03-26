import os, shutil

def cleanDirectory(path) :
	if os.path.exists(path) :
		shutil.rmtree(path)

	os.mkdir(path)