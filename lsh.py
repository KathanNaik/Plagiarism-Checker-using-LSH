#!/usr/bin/env python
# coding: utf-8

# # **IMPORT MODULES**

# In[332]:


import os
import pickle
import random
import numpy as np
from bs4 import BeautifulSoup


# ## **DEFINING CONSTANTS**

# In[333]:


# Location of the Dataset
ASSETS_LOCATION = "dataset"

# SHINGLING
SHINGLE_SIZE = 9  # Size of the shingles

# MIN HASHING
NO_OF_HASH_FUNCTIONS = 100  # Number of min hash functions we are using
HASH_MOD = 100003

# LOCALITY SENSITIVE HASHING (LSH)
CONSTANT_BAND = 20  # Number of Bands
CONSTANT_ROW = 2 # Number of Rows


# # **SHINGLING OPERATION FUNCTIONS**

# ## **GENERAL SHINGLING FUNCTIONS**

# ### **This function returns a set of k-shingles for the string passed as argument.**

# In[334]:


def findShingles(docData):
    shingles = []
    for i in range(0, len(docData) - SHINGLE_SIZE + 1):
        shingles.append(docData[i:i + SHINGLE_SIZE])
    
    return set(shingles) # set


# ## **SHINGLING OF CORPUS FUNCTIONS**

# ### **This function returns Vocabulary of shingles for entire corpus**

# In[335]:


def createShinglesVocab():
    dataFolder = os.listdir(ASSETS_LOCATION)
    dataFolder.sort()

    nameOfDocuments = []
    vocab = set()

    print("[.] Creating Shingles for the corpus!")

    for doc in dataFolder:
        fileName = ASSETS_LOCATION + f'/{doc}'
        nameOfDocuments.append(doc)
        filePtr = open(fileName, 'r', encoding='utf-8')
        filePtr = BeautifulSoup(filePtr, "html.parser")
        docData = filePtr.read()
        docVocab = findShingles(docData)
        filePtr.close()
        print(doc + ": File has been shingled successfully!")
        vocab.update(docVocab)
    

    print("[+] Shingles Vocabulary created successfully!")

    
    # filePtr = open("vocab.pkl", "wb")
    # pickle.dump(vocab, filePtr)

    # filePtr = open("nameOfDocuments.pkl", "wb")
    # pickle.dump(nameOfDocuments, filePtr)


    return nameOfDocuments, vocab # list, set
    


# ### **This Function assigns a unique id to each shingle of the Vocabulary**

# In[336]:


def assignIdToShingles(vocab):
    dictShinglesId = {}
    id = 0
    for shingle in vocab:
        dictShinglesId[shingle] = id
        id += 1

    return dictShinglesId # dict


# ### **Creating Shingles Matrix**

# In[337]:


def createShinglesMatrix(dictShinglesId):
    shingleMatrix = []
    
    print("[.] Creating Shingle Matrix...")
    dataFolder = os.listdir(ASSETS_LOCATION)
    dataFolder.sort()
    shingleMatrix = []

    for doc in dataFolder:
        fileName = ASSETS_LOCATION + f'/{doc}'
        filePtr = open(fileName, 'r', encoding='utf-8')
        filePtr = BeautifulSoup(filePtr, "html.parser")
        docData = filePtr.read()
        docVocab = findShingles(docData)
        filePtr.close()

        docRow = []

        for item in docVocab:
            docRow.append(dictShinglesId[item])
        
        docRow.sort()

        shingleMatrix.append(docRow)    

    print("[+] Shingle Matrix Created Successfully!")

    return shingleMatrix # Matrix (where row = number of docs and cols = shingles id)


# ## **SHINGLING OF QUERY**

# ### **Creating Query Matirx**

# In[338]:


def createQueryMatrix(userQuery, dictShinglesId):
    # Generating k-Shingles for the userQuery
    setQueryShingles = findShingles(userQuery) # set

    listQueryShinglesId = []

    for shingle in setQueryShingles:
        if shingle in dictShinglesId.keys():
            listQueryShinglesId.append(dictShinglesId[shingle])

    matrixQueryShinglesId = []
    listQueryShinglesId.sort()
    matrixQueryShinglesId.append(listQueryShinglesId)

    return matrixQueryShinglesId # matrix
    



# # **MIN HASH FUNCTIONS**

# ## **Generating Random Hash Functions**

# ### **This function generates {NO_OF_HASH_FUNCTIONS} random hash function**

# In[339]:


def generateRandomMinHashFunctions():
    hashFunctions = []

    random.seed(25)

    # Linear Hash Function =>> ax+b
    # coefficient = [a,b]
    for id in range(0, NO_OF_HASH_FUNCTIONS):
        a = random.randint(1, 100000)
        b = random.randint(1, 100000)
        coefficient = [a, b]

        hashFunctions.append(coefficient)
    
    return hashFunctions # list of min Hash functions


# # **Min Hash Technique Implementation**

# ## **Generating Signature Matrix**

# ### **Return Matrix of size nxm (n,m are passed as parameters) with all values as INF (very large)**

# In[340]:


def intitlizeMatrixWithInfinity(numberOfDocs):
    matrix = []
    for i in range(NO_OF_HASH_FUNCTIONS):
        row = []
        for j in range(numberOfDocs):
            row.append(HASH_MOD)  
        matrix.append(row) 
    
    return matrix


# ### **This function generates signature matrix by using min hashing technique using {NO_OF_HASH_FUNCTIONS}**

# In[341]:


def generateSignatureMatrix(shingleMatrix):
    # Stores Random Min Hash Fucntions
    hashFunction = generateRandomMinHashFunctions()

    numberOfDocs = len(shingleMatrix)
    print(numberOfDocs)
    # Initilizing all the rows and cols of signatureMatrix with INFINITY (Very Large Value)
    signatureMatrix = intitlizeMatrixWithInfinity(numberOfDocs)

    # Min Hash Algorithm
    print("[.] Processing Signature Matrix...")
    for i in range(len(shingleMatrix)):
        for j in range(NO_OF_HASH_FUNCTIONS):
            # a and b are the constants of the min hash function
            a = hashFunction[j][0]
            b = hashFunction[j][1]

            for k in shingleMatrix[i]:
                # a * x + b is the hash function where a and b are constants we generated during random min hash function generator while x is a variable which take the value to be hashed.
                hashKey = ((a * (k + 1)) + b) % HASH_MOD
                if hashKey < signatureMatrix[j][i]:
                    signatureMatrix[j][i] = hashKey

    print("[+] Signature Matrix Creating Successfull")

    '''
    Signature Matrix is a NO_OF_HASH_FUNCTION x NO_OF_DOCUMENTS sized matrix where each cell stores the hash value.
    '''

    return signatureMatrix # Matrix (rows = number of hash functions & cols = no of docs)


# # **LSH: Locality Sensitive Hashing**

# ### **LSH Implementation Function**

# In[342]:


def lsh(signatureMatrix):
    '''
    Input: Signature Matrix, number of bands & number of rows.
    We perform LSH on the signature matrix by divinding the signature matrix in b bands where each bans contains r rows.    
    '''

    print("[.] LSH of Signature Matrix Started...")

    bucketForBands = {} # bucket (dictinory) that stores sub buckets for all band
    numberOfDocuments = len(signatureMatrix[0])

    for bandB in range(0, CONSTANT_BAND):
        bucketForBandB = {}

        for docNumber in range(numberOfDocuments):
            hashVector = []
            try:
                hashVector = [signatureMatrix[row][docNumber] for row in range (bandB * CONSTANT_ROW, ((bandB + 1) * CONSTANT_ROW))]
            except:
                hashVector = [signatureMatrix[row][docNumber] for row in range (bandB * CONSTANT_ROW, ((bandB) * CONSTANT_ROW))]
                pass # I passed this statement and didn't wrote anything
        
            bucketId = "".join(map(str, hashVector))

            if not bucketForBandB.get(bucketId):
                bucketForBandB[bucketId] = set()

            bucketForBandB[bucketId].add(docNumber)
        bucketForBands[bandB] = bucketForBandB

    print("[+] LSH Created Successfully")

    return bucketForBands # dict
    


# ### **Function to perform LSH on CORPUS**

# In[343]:


def performLSHcorpus():
    '''
    This is the function that call the relevant functions to perform LSH on the CORPUS of data. The Steps involved are:

    1. Creating Shingles Vocabulary
    2. Creating Shingles Matrix
    3. Creating Signature Matrix using Min Hashing Technique
    4. Performing LSH on Signature Matrix

    This function return the bukcet formed by the LSH Function on corpus.
    '''

    # 1. Creating Shingles Vocabulary
    nameOfDocuments, vocab = createShinglesVocab()

    # 2. Creating Shingles Matrix
    dictShinglesId = assignIdToShingles(vocab)
    shingleMatrix = createShinglesMatrix(dictShinglesId)
    # print(shingleMatrix)

    # 3. Creating Signature Matrix
    signatureMatrix = generateSignatureMatrix(shingleMatrix)
    # print(signatureMatrix)

    # 4. Performing LSH on Signature Matrix
    corpusBucket = lsh(signatureMatrix)
    print(corpusBucket)

    return corpusBucket, dictShinglesId



# In[344]:


# corpusBucket, dictShinglesId = performLSHcorpus()


# ### **Function to perform LSH on Query**

# In[345]:


def performLSHquery(query, dictShinglesId):
    '''
    This function takes the query from the user and return the bukcet formed by the LSH Function on query. The Steps involved are:

    1. Creating Shingles Vocabulary
    2. Creating Shingles Matrix
    3. Creating Signature Matrix using Min Hashing Technique
    4. Performing LSH on Signature Matrix

    This function return the bukcet formed by the LSH Function on query.
    '''

    # 1. Creating Shingles Vocabulary
    # Already done while performing shingling of corpus 
    # Result in vocab set

    # 2. Creating Shingles Matrix
    shingleMatrix = createQueryMatrix(query, dictShinglesId)

    # 3. Creating Signature Matrix
    signatureMatrix = generateSignatureMatrix(shingleMatrix)

    # 4. Performing LSH on Signature Matrix
    queryBucket = lsh(signatureMatrix)


    return queryBucket


# 

# ## **INPUT QUERY FROM USER**

# In[346]:


# query = input("Enter your Query: ")
query = ""
print(query)
# matrixQueryShinglesId = createQueryMatrix(query, dictShinglesId)


# In[347]:


# matrixQueryShinglesId


# In[348]:


# queryBucket = performLSHquery(query, dictShinglesId)


# In[349]:


# queryBucket


# In[350]:


# corpusBucket


# # **Functions for Generating OUTPUT for User's Query**

# ### **Retriving Data of a specific document by using Doc ID**

# In[351]:


def getDataForDocumnetById(docId):
    dataFolder = os.listdir(ASSETS_LOCATION)
    dataFolder.sort()
    
    docName = dataFolder[docId]
    docLocation = ASSETS_LOCATION + f'/{docName}'
    filePtr = open(docLocation, 'r')
    filePtr = BeautifulSoup(filePtr, "html.parser")
    docData = filePtr.read()

    return docName, docData


# ### **Finding similar document**

# In[277]:


# def findSimilarDocs(corpusBucket ,queryBucket):
#     """
#     This Function return the set of all the documents that are similar to the query input by the user
#     """
#
#     similarDocs = set()
#
#     for queryBand in queryBucket.keys():
#         for (queryBucketIndex, queryBucketDocs) in queryBucket[queryBand].items():
#             if queryBucketDocs:


# In[352]:


def find_similar_docs(query_buckets, docs_buckets):
    """
    Given the `docs_buckets` and the buckets `query_buckets` formed by the query
    finds all the similar documents to the ones in `query_buckets` in `docs_buckets`.
    """
    similar_docs = set()

    for q_band_key in query_buckets.keys():
        for q_bucket_idx, q_bucket_docs in query_buckets[q_band_key].items():
            if q_bucket_docs:
                if (
                    q_band_key in docs_buckets
                    and q_bucket_idx in docs_buckets[q_band_key]
                ):
                    similar_docs.update(docs_buckets[q_band_key][q_bucket_idx])
    return similar_docs


# In[353]:


# qb = {
#     1: {1: [0], 3: [0]},
#     2: {4: [0]},
# }

# db = {
#     1: {1: [0, 1, 2], 2: [3, 4]},
#     2: {4: [6]}
# }
# docIdList = find_similar_docs(queryBucket, corpusBucket)


# In[354]:


# docIdList


# In[355]:


# for item in docIdList:
#     print(getDataForDocumnetById(item))


# In[ ]:








