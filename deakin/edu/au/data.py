# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2021.  Mohamed Reda Bouadjenek, Deakin University                       +
#               Email:  reda.bouadjenek@deakin.edu.au                                    +
#                                                                                        +
#       Licensed under the Apache License, Version 2.0 (the "License");                  +
#       you may not use this file except in compliance with the License.                 +
#       You may obtain a copy of the License at:                                         +
#                                                                                        +
#       http://www.apache.org/licenses/LICENSE-2.0                                       +
#                                                                                        +
#       Unless required by applicable law or agreed to in writing, software              +
#       distributed under the License is distributed on an "AS IS" BASIS,                +
#       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.         +
#       See the License for the specific language governing permissions and              +
#       limitations under the License.                                                   +
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
from keras.datasets import cifar100
from graphviz import Digraph
import tensorflow as tf
import pandas as pd
import keras


# mapping_fine_to_cluster = {0: 5, 1: 2, 2: 3, 3: 6, 4: 6, 5: 0, 6: 2, 7: 2, 8: 8, 9: 1, 10: 1, 11: 3, 12: 8,
#                                13: 8, 14: 2, 15: 6, 16: 1, 17: 8, 18: 2, 19: 6, 20: 0, 21: 3, 22: 1, 23: 7, 24: 2,
#                                25: 0, 26: 2, 27: 6, 28: 1, 29: 6, 30: 6, 31: 6, 32: 6, 33: 4, 34: 6, 35: 3, 36: 6,
#                                37: 8, 38: 6, 39: 1, 40: 1, 41: 8, 42: 6, 43: 6, 44: 2, 45: 2, 46: 3, 47: 4, 48: 8,
#                                49: 7, 50: 6, 51: 5, 52: 4, 53: 5, 54: 5, 55: 6, 56: 4, 57: 5, 58: 8, 59: 4, 60: 7,
#                                61: 1, 62: 5, 63: 6, 64: 6, 65: 6, 66: 6, 67: 6, 68: 7, 69: 1, 70: 5, 71: 7, 72: 6,
#                                73: 2, 74: 6, 75: 6, 76: 1, 77: 2, 78: 2, 79: 2, 80: 6, 81: 8, 82: 2, 83: 5, 84: 0,
#                                85: 8, 86: 1, 87: 0, 88: 6, 89: 8, 90: 8, 91: 2, 92: 5, 93: 6, 94: 0, 95: 6, 96: 4,
#                                97: 6, 98: 3, 99: 2}

cifar100_mapping_coarse_to_top = {0: 0, 1: 0, 2: 0, 3: 1, 4: 0, 5: 1, 6: 1, 7: 0, 8: 0, 9: 1, 10: 1, 11: 0, 12: 0,
                                  13: 0, 14: 0,
                                  15: 0, 16: 0, 17: 0, 18: 1, 19: 1}


def map_fine_to_cluster_cifar100(y, mapping):
    """
    This function is only used to create label for clusters if used.  Clusters are obtained from:

    :param y:
    :type y:
    :return:
    :rtype:
    """
    # Mapping fine -> cluster

    y_top = []
    for f in y:
        k = f[0]
        c = np.array([mapping[k]])
        y_top.append(c)
    return np.array(y_top)


class Cifar100:
    LABELS = [['bio organism', 'objects'],
              ['aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetales',
               'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores',
               'large_man-made_outdoor_things', 'large_natural_outdoor_scenes', 'large_omnivores_and_herivores',
               'medium_mammals', 'non-insect_inverterates', 'people', 'reptiles', 'small_mammals', 'trees',
               'vehicles_1', 'vehicles_2'],
              ['apple', 'aquarium_fish', 'ray', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'owl', 'boy',
               'ridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee',
               'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant',
               'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyoard', 'lamp', 'lawn_mower',
               'leopard', 'lion', 'lizard', 'loster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
               'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
               'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rait', 'raccoon', 'ray', 'road', 'rocket', 'rose',
               'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel',
               'streetcar', 'sunflower', 'sweet_pepper', 'tale', 'tank', 'telephone', 'television', 'tiger', 'tractor',
               'train', 'trout', 'tulip', 'turtle', 'wardroe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']]

    def __init__(self):
        """

        :param type: to indicate if to use coarse classes as given in the cifar100 dataset or use clusters.
        :type type: str
        """
        (X_c_train, y_c_train), (X_c_test, y_c_test) = cifar100.load_data(label_mode='coarse')
        (X_f_train, y_f_train), (X_f_test, y_f_test) = cifar100.load_data(label_mode='fine')

        y_top_train = map_fine_to_cluster_cifar100(y_c_train, cifar100_mapping_coarse_to_top)
        y_top_test = map_fine_to_cluster_cifar100(y_c_test, cifar100_mapping_coarse_to_top)

        self.X_train = X_f_train
        self.X_val = X_f_test[:5000]
        self.X_test = X_f_test[5000:]

        self.y_train = [y_top_train, y_c_train, y_f_train]
        self.y_val = [y_top_test[:5000], y_c_test[:5000], y_f_test[:5000]]
        self.y_test = [y_top_test[5000:], y_c_test[5000:], y_f_test[5000:]]

        self.image_size = self.X_train[0].shape

        self.num_classes_l0 = len(set([v[0] for v in y_top_train]))
        self.num_classes_l1 = len(set([v[0] for v in y_c_train]))
        self.num_classes_l2 = len(set([v[0] for v in y_f_train]))

        # Encoding the taxonomy
        m0 = [[0 for x in range(self.num_classes_l1)] for y in range(self.num_classes_l0)]
        for (t, c) in zip(y_top_train, y_c_train):
            t = t[0]
            c = c[0]
            m0[t][c] = 1

        m1 = [[0 for x in range(self.num_classes_l2)] for y in range(self.num_classes_l1)]
        for (t, c) in zip(y_c_train, y_f_train):
            t = t[0]
            c = c[0]
            m1[t][c] = 1
        self.taxonomy = [m0, m1]

    def draw_taxonomy(self):
        """
        This method draws the taxonomy using the graphviz library.

        :return:
        :rtype: Digraph
        """
        u = Digraph('unix', filename='diagram8',
                    node_attr={'color': 'lightblue2', 'style': 'filled'}, strict=True)
        u.attr(size='6,6')
        u.attr(rankdir="LR")

        for i in range(len(self.taxonomy[0])):
            u.edge('root', self.LABELS[0][i], self.LABELS[0][i])

        for l in range(len(self.taxonomy)):
            for i in range(len(self.taxonomy[l])):
                for j in range(len(self.taxonomy[l][i])):
                    if self.taxonomy[l][i][j] == 1:
                        u.edge(self.LABELS[l][i], self.LABELS[l + 1][j])

        return u
    
def load_dataset(path,images_path,image_size):


    df=pd.read_csv(path)
    
    global imagesize
    
    imagesize= image_size
        
    
    filenames=df['fname'].values
    labels_3=df['class1'].values
    labels_2=df['class2'].values
    labels_1=df['class3'].values
    
    
    paths=np.repeat(images_path,filenames.size,axis=0)

    dataset = tf.data.Dataset.from_tensor_slices((paths,filenames,labels_1,labels_2,labels_3))
    dataset = dataset.map(_parse_function,num_parallel_calls=8)
    
    n_elements= len(df)
    
    
    dataset = dataset.shuffle(buffer_size = n_elements,seed=tf.random.set_seed(1234))
    
    # 4. batch
    dataset = dataset.batch(n_elements, drop_remainder=False)
  
    # 6. prefetch
    dataset = dataset.prefetch(1)
    
    iterator = iter(dataset)

    X, lab1,lab2,lab3 = iterator.get_next()
    
    X=X.numpy()
    
    lab1=lab1.numpy()
    
    lab2=lab2.numpy()
    
    lab3=lab3.numpy()
    
    return X,lab1,lab2,lab3


def _parse_function(paths,filenames, label_1,label_2,labels_3):
    
    image_string = tf.io.read_file(paths+'\\'+filenames)
    img = tf.image.decode_jpeg(image_string,channels=imagesize[2])
    img=tf.image.resize(img,[imagesize[0],imagesize[1]])
    
    return img, label_1,label_2,labels_3

def taxonomy(num_classes_l0,num_classes_l1,num_classes_l2,lab1,lab2,lab3):
    
    m0 = [[0 for x in range(num_classes_l1)] for y in range(num_classes_l0)]

    for (t, c) in zip(lab1, lab2):
        m0[t][c] = 1

    m1 = [[0 for x in range(num_classes_l2)] for y in range(num_classes_l1)]

    for (t, c) in zip(lab2, lab3):
        m1[t][c] = 1

    
    taxonomy = [m0, m1]
        
    return taxonomy
        
def draw_taxonomy(taxonomy,LABELS):
   """
   This method draws the taxonomy using the graphviz library.
   :return:
   :rtype: Digraph
    """
   u = Digraph('unix', filename='diagram8',node_attr={'color': 'lightblue2', 'style': 'filled'}, strict=True)
   u.attr(size='6,6')
   u.attr(rankdir="LR")

   for i in range(len(taxonomy[0])):
       u.edge('root', LABELS[0][i], LABELS[0][i])

   for l in range(len(taxonomy)):
       for i in range(len(taxonomy[l])):
           for j in range(len(taxonomy[l][i])):
               if taxonomy[l][i][j] == 1:
                   u.edge(LABELS[l][i], LABELS[l + 1][j])

   return u 

class Standford_Cars:
    
    LABELS = [['Hatchback',	'Sedan','Crew Cab',	'SUV',	'Convertible','Coupe','Wagon','Hatchback Coupe','Van','Minivan','Extended Cab',	'Regular Cab',	'Coupe Convertible'],
              
              ['Coupe Convertible',	'Acura Hatchback',	'Acura Sedan',	'AM General Crew Cab',	'AM General SUV',	'Aston Convertible',	'Aston Coupe',	'Audi Sedan',	'Audi Wagon',	
               'Audi Coupe',	'Audi Convertible',	'Audi Hatchback',	'Bentley Sedan',	'Bentley Coupe',	'Bentley Convertible',	
               'BMW Convertible',	'BMW Coupe',	'BMW Sedan',	'BMW Wagon',	'BMW SUV',	'Bugatti Convertible',	'Bugatti Coupe','Buick SUV',	'Buick Sedan',	'Cadillac Sedan',	
               'Cadillac Crew Cab',	'Cadillac SUV',	'Chevrolet Crew Cab',	'Chevrolet Convertible',	'Chevrolet Coupe',	'Chevrolet Hatchback Coupe',	'Chevrolet Van',	'Chevrolet Minivan',	
               'Chevrolet Sedan',	'Chevrolet Extended Cab',	'Chevrolet Regular Cab',	'Chevrolet SUV',	'Chevrolet Wagon',	'Chrysler Sedan',	'Chrysler SUV',	'Chrysler Convertible',	
               'Chrysler Minivan',	'Daewoo Wagon',	'Dodge Wagon',	'Dodge Minivan',	'Dodge Coupe',	'Dodge Sedan',	'Dodge Extended Cab',	'Dodge Crew Cab',	'Dodge SUV',	'Dodge Van',	
               'Eagle Hatchback',	'Ferrari Convertible',	'Ferrari Coupe',	'FIAT Hatchback',	'FIAT Convertible',	'Fisker Sedan',	'Ford SUV',	'Ford Van',	'Ford Regular Cab',	'Ford Crew Cab',	
               'Ford Sedan',	'Ford Minivan',	'Ford Coupe',	'Ford Convertible','Ford Extended Cab',	'Geo Convertible',	'GMC SUV',	'GMC Extended Cab',	'GMC Van',	'Honda Coupe',	'Honda Sedan',	
               'Honda Minivan',	'Hyundai Sedan',	'Hyundai Hatchback',	'Hyundai SUV',	'Infiniti Coupe',	'Infiniti SUV',	'Isuzu SUV',	'Jaguar Hatchback Coupe',	'Jeep SUV',	'Lamborghini Coupe',	
               'Land Rover SUV',	'Lincoln Sedan',	'Maybach Convertible',	'Mazda SUV',	'McLaren Coupe',	'Mercedes-Benz Convertible',	'Mercedes-Benz Sedan',	'Mercedes-Benz Coupe',	'Mercedes-Benz Van',	
               'MINI Convertible',	'Mitsubishi Sedan',	'Nissan Coupe',	'Nissan Hatchback',	'Nissan Van',	'Plymouth Coupe',	'Porsche Sedan',	'Ram Minivan',	'Rolls-Royce Sedan',	'Rolls-Royce Coupe Convertible',	
               'Scion Hatchback',	'Smart Convertible',	'Spyker Convertible',	'Spyker Coupe',	'Suzuki Sedan',	'Suzuki Hatchback',	'Tesla Sedan',	'Toyota SUV',	'Toyota Sedan',	'Volkswagen Hatchback',	'Volvo Sedan',	
               'Volvo Hatchback',	'Volvo SUV'],
               
              ['Acura Integra Type R 2001',	'Acura RL Sedan 2012',	'Acura TL Sedan 2012',	'Acura TL Type-S 2008',	'Acura TSX Sedan 2012',	'Acura ZDX Hatchback 2012',	'AM General  HUMMER H2 SUT Crew Cab 2009',	
               'AM General HUMMER H3T Crew Cab 2010',	'AM General Hummer SUV 2000',	'Aston Martin V8 Vantage Convertible 2012',	'Aston Martin V8 Vantage Coupe 2012',	'Aston Martin Virage Convertible 2012',	
               'Aston Martin Virage Coupe 2012',	'Audi 100 Sedan 1994',	'Audi 100 Wagon 1994',	'Audi A5 Coupe 2012',	'Audi R8 Coupe 2012',	'Audi RS 4 Convertible 2008',	'Audi S4 Sedan 2007',	'Audi S4 Sedan 2012',	
               'Audi S5 Convertible 2012',	'Audi S5 Coupe 2012',	'Audi S6 Sedan 2011',	'Audi TT Hatchback 2011',	'Audi TT RS Coupe 2012',	'Audi TTS Coupe 2012',	'Audi V8 Sedan 1994',	'Bentley Arnage Sedan 2009',	
               'Bentley Continental Flying Spur Sedan 2007',	'Bentley Continental GT Coupe 2007',	'Bentley Continental GT Coupe 2012',	'Bentley Continental Supersports Conv. Convertible 2012',	
               'Bentley Mulsanne Sedan 2011',	'BMW 1 Series Convertible 2012',	'BMW 1 Series Coupe 2012',	'BMW 3 Series Sedan 2012',	'BMW 3 Series Wagon 2012',	'BMW 6 Series Convertible 2007',	
               'BMW ActiveHybrid 5 Sedan 2012',	'BMW M3 Coupe 2012',	'BMW M5 Sedan 2010',	'BMW M6 Convertible 2010',	'BMW X3 SUV 2012',	'BMW X5 SUV 2007',	'BMW X6 SUV 2012',	'BMW Z4 Convertible 2012',	
               'Bugatti Veyron 16.4 Convertible 2009',	'Bugatti Veyron 16.4 Coupe 2009',	'Buick Enclave SUV 2012',	'Buick Rainier SUV 2007',	'Buick Regal GS 2012',	'Buick Verano Sedan 2012',	
               'Cadillac CTS-V Sedan 2012',	'Cadillac Escalade EXT Crew Cab 2007',	'Cadillac SRX SUV 2012',	'Chevrolet Avalanche Crew Cab 2012',	'Chevrolet Camaro Convertible 2012',	'Chevrolet Cobalt SS 2010',	
               'Chevrolet Corvette Convertible 2012',	'Chevrolet Corvette Ron Fellows Edition Z06 2007',	'Chevrolet Corvette ZR1 2012',	'Chevrolet Express Cargo Van 2007',	'Chevrolet Express Van 2007',	'Chevrolet HHR SS 2010',
               'Chevrolet Impala Sedan 2007',	'Chevrolet Malibu Hybrid Sedan 2010',	'Chevrolet Malibu Sedan 2007',	'Chevrolet Monte Carlo Coupe 2007',	'Chevrolet Silverado 1500 Classic Extended Cab 2007',	
               'Chevrolet Silverado 1500 Extended Cab 2012',	'Chevrolet Silverado 1500 Hybrid Crew Cab 2012',	'Chevrolet Silverado 1500 Regular Cab 2012',	'Chevrolet Silverado 2500HD Regular Cab 2012',	
               'Chevrolet Sonic Sedan 2012',	'Chevrolet Tahoe Hybrid SUV 2012',	'Chevrolet TrailBlazer SS 2009',	'Chevrolet Traverse SUV 2012',	'Chrysler 300 SRT-8 2010',	'Chrysler Aspen SUV 2009',	
               'Chrysler Crossfire Convertible 2008',	'Chrysler PT Cruiser Convertible 2008',	'Chrysler Sebring Convertible 2010',	'Chrysler Town and Country Minivan 2012',	'Daewoo Nubira Wagon 2002',	'Dodge Caliber Wagon 2007',	
               'Dodge Caliber Wagon 2012',	'Dodge Caravan Minivan 1997',	'Dodge Challenger SRT8 2011',	'Dodge Charger Sedan 2012',	'Dodge Charger SRT-8 2009',	'Dodge Dakota Club Cab 2007',	'Dodge Dakota Crew Cab 2010',	
               'Dodge Durango SUV 2007',	'Dodge Durango SUV 2012',	'Dodge Journey SUV 2012',	'Dodge Magnum Wagon 2008',	'Dodge Ram Pickup 3500 Crew Cab 2010',	'Dodge Ram Pickup 3500 Quad Cab 2009',	'Dodge Sprinter Cargo Van 2009',	
               'Eagle Talon Hatchback 1998',	'Ferrari 458 Italia Convertible 2012',	'Ferrari 458 Italia Coupe 2012',	'Ferrari California Convertible 2012',	'Ferrari FF Coupe 2012',	'FIAT 500 Abarth 2012',	'FIAT 500 Convertible 2012',	
               'Fisker Karma Sedan 2012',	'Ford Edge SUV 2012',	'Ford E-Series Wagon Van 2012',	'Ford Expedition EL SUV 2009',	'Ford F-150 Regular Cab 2007',	'Ford F-150 Regular Cab 2012',	'Ford F-450 Super Duty Crew Cab 2012',	'Ford Fiesta Sedan 2012',	
               'Ford Focus Sedan 2007',	'Ford Freestar Minivan 2007',	'Ford GT Coupe 2006',	'Ford Mustang Convertible 2007',	'Ford Ranger SuperCab 2011',	'Geo Metro Convertible 1993',	'GMC Acadia SUV 2012',	'GMC Canyon Extended Cab 2012',	
               'GMC Savana Van 2012',	'GMC Terrain SUV 2012',	'GMC Yukon Hybrid SUV 2012',	'Honda Accord Coupe 2012',	'Honda Accord Sedan 2012',	'Honda Odyssey Minivan 2007',	'Honda Odyssey Minivan 2012',	'Hyundai Accent Sedan 2012',	
               'Hyundai Azera Sedan 2012',	'Hyundai Elantra Sedan 2007',	'Hyundai Elantra Touring Hatchback 2012',	'Hyundai Genesis Sedan 2012',	'Hyundai Santa Fe SUV 2012',	'Hyundai Sonata Hybrid Sedan 2012',	'Hyundai Sonata Sedan 2012',	
               'Hyundai Tucson SUV 2012',	'Hyundai Veloster Hatchback 2012',	'Hyundai Veracruz SUV 2012',	'Infiniti G Coupe IPL 2012',	'Infiniti QX56 SUV 2011',	'Isuzu Ascender SUV 2008',	'Jaguar XK XKR 2012',	'Jeep Compass SUV 2012',	
               'Jeep Grand Cherokee SUV 2012',	'Jeep Liberty SUV 2012',	'Jeep Patriot SUV 2012',	'Jeep Wrangler SUV 2012',	'Lamborghini Aventador Coupe 2012',	'Lamborghini Diablo Coupe 2001',	'Lamborghini Gallardo LP 570-4 Superleggera 2012',	
               'Lamborghini Reventon Coupe 2008',	'Land Rover LR2 SUV 2012',	'Land Rover Range Rover SUV 2012',	'Lincoln Town Car Sedan 2011',	'Maybach Landaulet Convertible 2012',	'Mazda Tribute SUV 2011',	'McLaren MP4-12C Coupe 2012',	
               'Mercedes-Benz 300-Class Convertible 1993',	'Mercedes-Benz C-Class Sedan 2012',	'Mercedes-Benz E-Class Sedan 2012',	'Mercedes-Benz S-Class Sedan 2012',	'Mercedes-Benz SL-Class Coupe 2009',	'Mercedes-Benz Sprinter Van 2012',	
               'MINI Cooper Roadster Convertible 2012',	'Mitsubishi Lancer Sedan 2012',	'Nissan 240SX Coupe 1998',	'Nissan Juke Hatchback 2012',	'Nissan Leaf Hatchback 2012',	'Nissan NV Passenger Van 2012',	'Plymouth Neon Coupe 1999',	
               'Porsche Panamera Sedan 2012',	'Ram C/V Cargo Van Minivan 2012',	'Rolls-Royce Ghost Sedan 2012',	'Rolls-Royce Phantom Drophead Coupe Convertible 2012',	'Rolls-Royce Phantom Sedan 2012',	'Scion xD Hatchback 2012',	
               'smart fortwo Convertible 2012',	'Spyker C8 Convertible 2009',	'Spyker C8 Coupe 2009',	'Suzuki Aerio Sedan 2007',	'Suzuki Kizashi Sedan 2012',	'Suzuki SX4 Hatchback 2012',	'Suzuki SX4 Sedan 2012',	
               'Tesla Model S Sedan 2012',	'Toyota 4Runner SUV 2012',	'Toyota Camry Sedan 2012',	'Toyota Corolla Sedan 2012',	'Toyota Sequoia SUV 2012',	'Volkswagen Beetle Hatchback 2012',	'Volkswagen Golf Hatchback 1991',	
               'Volkswagen Golf Hatchback 2012',	'Volvo 240 Sedan 1993',	'Volvo C30 Hatchback 2012',	'Volvo XC90 SUV 2007']]
    
    def __init__(self,image_size): 
        
        self.image_size=image_size
        
        
        train_data_url='http://ai.stanford.edu/~jkrause/car196/car_ims.tgz'
        
        filename='car_ims.tgz'
        
        print('Preparing dataset..')
        
        dataset_path=keras.utils.get_file(filename,train_data_url,extract=True,untar=True)
        

        train_csv_url='https://docs.google.com/uc?export=download&id=1BPR6qoSr3o1J670NsS_cST31p2Ai3N54'
        train_path=keras.utils.get_file("train_cars.csv", train_csv_url)
    
        
        test_csv_url='https://docs.google.com/uc?export=download&id=1enfdXAi7w93iRz2xDRsu21OOfTxngwCv'
        test_path=keras.utils.get_file("test_cars.csv", test_csv_url)
        
        destDir=dataset_path[0:-len(filename)]+'car_ims'
    
        X,lab1,lab2,lab3 = load_dataset(path=train_path,images_path=destDir,image_size=self.image_size)
        X_test,lab1t,lab2t,lab3t = load_dataset(path=test_path,images_path=destDir,image_size=self.image_size)
    
        
        self.X_train=X
        
        self.X_test=X_test[2500:]
        
        self.X_val=X_test[:2500]

        self.y_train = [lab1, lab2, lab3]
        
        self.y_val = [lab1t[:2500], lab2t[:2500], lab3t[:2500]]
        
        self.y_test = [lab1t[2500:], lab2t[2500:], lab3t[2500:]]
        
        #self.image_size = self.X_train[0].shape
        
        y,idx=tf.unique(lab1)
        self.num_classes_l0 = tf.size(y).numpy()
        y,idx=tf.unique(lab2)
        self.num_classes_l1 = tf.size(y).numpy()
        y,idx=tf.unique(lab3)
        self.num_classes_l2 = tf.size(y).numpy()        

    
        self.taxonomy = taxonomy(self.num_classes_l0,self.num_classes_l1,self.num_classes_l2,lab1,lab2,lab3)
        
        self.draw_taxonomy= draw_taxonomy(self.taxonomy,self.LABELS)





if __name__ == '__main__':
    dataset = Standford_Cars(image_size=(32,32,3))
    print(dataset.num_classes_l0)
    print(dataset.num_classes_l1)
    print(dataset.num_classes_l2)
    print(dataset.taxonomy)
