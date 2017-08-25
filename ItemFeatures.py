import csv
import pickle
import pandas as pd
import numpy as np


path = "./data/ml-1m/movies.dat"
path = "./data/ml-100k/u.item"


def getGenre():
    f = open(path ,'r')
    count = 0
    genre_list = [None] * 3952
    genre_vectors = [None] * 3952

    for line in f:
        l = line.split("::")
        count += 1
        genre = l[2].strip().split('|')
        for i in range(len(genre)):
            if genre[i] != '':
                genre_list.append(genre[i])
    genre_set = set(genre_list)
    genre_result = list(genre_set)
    genre_count = len(genre_set)
    f.close()
    f = open(path,'r')
    for line in f:
        temp = [0.0] * genre_count
        l = line.split("::")
        genre = l[2].strip().split('|')
        for i in range(len(genre)):
            if genre[i] != '':
                genre_list.append(genre[i])
                temp[genre_result.index(genre[i])] = 1
        genre_vectors[int(l[0]) - 1] = temp
    f.close()


    return genre_vectors

def getYear():
    f = open(path ,'r')
    year_list = [None] * 3952

    for line in f:
        l = line.split("::")
        temp = l[1].split("(")
        id = int(l[0]) - 1
        if len(temp) == 5:
            year = l[1].split("(")[4].strip(")")
        elif len(temp) == 4:
            year = l[1].split("(")[3].strip(")")
        elif len(temp) ==  3:
            year = l[1].split("(")[2].strip(")")
        else:
            year = l[1].split("(")[1].strip(")")
        year_list[id] =  int(year)

    year_vectors = [None] * 3952
    for id, y in enumerate(year_list):
        if y is not None:
            rate = (y-1900) / 7
            number = 1.0
            rate_vector = [0.0] * 20
            if rate >= 0 and rate <= 0.5:
                rate_vector[0] = number
            elif rate > 0.5 and rate <= 1.0:
                rate_vector[1] = number
            elif rate > 1.0 and rate <= 1.5:
                rate_vector[2] = number
            elif rate > 1.5 and rate <= 2.0:
                rate_vector[3] = number
            elif rate > 2.0 and rate <= 2.5:
                rate_vector[4] = number
            elif rate > 2.5 and rate <= 3.0:
                rate_vector[5] = number
            elif rate > 3.0 and rate <= 3.5:
                rate_vector[6] = number
            elif rate > 3.5 and rate <= 4.0:
                rate_vector[7] = number
            elif rate > 4.0 and rate <= 4.5:
                rate_vector[8] = number
            elif rate > 4.5 and rate <= 5.0:
                rate_vector[9] = number
            elif rate > 5.0 and rate <= 5.5:
                rate_vector[10] = number
            elif rate > 5.5 and rate <= 6.0:
                rate_vector[11] = number
            elif rate > 6.0 and rate <= 6.5:
                rate_vector[12] = number
            elif rate > 6.5 and rate <= 7.0:
                rate_vector[13] = number
            elif rate > 7.0 and rate <= 7.5:
                rate_vector[14] = number
            elif rate > 7.5 and rate <= 8.0:
                rate_vector[15] = number
            elif rate > 8.0 and rate <= 8.5:
                rate_vector[16] = number
            elif rate > 8.5 and rate <= 9.0:
                rate_vector[17] = number
            elif rate > 9.0 and rate <= 9.5:
                rate_vector[18] = number
            elif rate > 9.5 and rate <= 10.0:
                rate_vector[19] = number
            year_vectors[id] = rate_vector

    return year_vectors

def getYear100k():
    f = open(path ,'r')
    year_list = [None] * 1682

    for line in f:
        l = line.split("|")
        temp = l[1].split("(")
        id = int(l[0]) - 1
        if len(temp) == 5:
            year = l[1].split("(")[4].strip(")")
        elif len(temp) == 4:
            year = l[1].split("(")[3].strip(")")
        elif len(temp) ==  3:
            year = l[1].split("(")[2].strip(")")
        else:
            year = l[1].split("(")[1].strip(")")
        year_list[id] =  int(year)

    year_vectors = [None] * 1682
    for id, y in enumerate(year_list):
        if y is not None:
            rate = (y-1900) / 7
            number = 1.0
            rate_vector = [0.0] * 20
            if rate >= 0 and rate <= 0.5:
                rate_vector[0] = number
            elif rate > 0.5 and rate <= 1.0:
                rate_vector[1] = number
            elif rate > 1.0 and rate <= 1.5:
                rate_vector[2] = number
            elif rate > 1.5 and rate <= 2.0:
                rate_vector[3] = number
            elif rate > 2.0 and rate <= 2.5:
                rate_vector[4] = number
            elif rate > 2.5 and rate <= 3.0:
                rate_vector[5] = number
            elif rate > 3.0 and rate <= 3.5:
                rate_vector[6] = number
            elif rate > 3.5 and rate <= 4.0:
                rate_vector[7] = number
            elif rate > 4.0 and rate <= 4.5:
                rate_vector[8] = number
            elif rate > 4.5 and rate <= 5.0:
                rate_vector[9] = number
            elif rate > 5.0 and rate <= 5.5:
                rate_vector[10] = number
            elif rate > 5.5 and rate <= 6.0:
                rate_vector[11] = number
            elif rate > 6.0 and rate <= 6.5:
                rate_vector[12] = number
            elif rate > 6.5 and rate <= 7.0:
                rate_vector[13] = number
            elif rate > 7.0 and rate <= 7.5:
                rate_vector[14] = number
            elif rate > 7.5 and rate <= 8.0:
                rate_vector[15] = number
            elif rate > 8.0 and rate <= 8.5:
                rate_vector[16] = number
            elif rate > 8.5 and rate <= 9.0:
                rate_vector[17] = number
            elif rate > 9.0 and rate <= 9.5:
                rate_vector[18] = number
            elif rate > 9.5 and rate <= 10.0:
                rate_vector[19] = number
            year_vectors[id] = rate_vector

    return year_vectors


def save(genres,year_vectors):
    count = len(genres)
    res = []
    for i in range(count):
        res.append(genres[i])
    with open('./data/1m-item-features.csv', 'wb') as fp:
        pickle.dump(res, fp)
    return res

def getItemFeatures():
    genres = getGenre()
    print(len(genres))
    year_vectors = getYear()
    print(len(year_vectors))
    count = len(genres)
    res = [None] * 3952
    for i in range(count):
        #print(genres[i])
        if genres[i] is None and year_vectors[i] is None:
            continue
        else:
            res[i] = genres[i] + year_vectors[i] # 48
    print(len(res))
    return res


def getItemFeatures100k():
    genres = getGenreVector(p=path)
    year_vectors = getYear100k()
    count = len(genres)
    res = [None] * 1682
    for i in range(count):
        if genres[i] is None and year_vectors[i] is None:
            continue
        else:
            res[i] = genres[i] + year_vectors[i] # 48
    return res

def getGenreVector(p="B:/Datasets/MovieLens/ml-100k/u.item"):
    genres = []
    f = open(p,'r')
    for line in f:
        l = line.split("|")
        genres.append([float(i) for i in l[5:]])
    f.close()

    return genres

if __name__ == '__main__':
    #save(getGenre(),getYear())#18 + 20
    getItemFeatures100k()

