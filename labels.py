from collections import namedtuple

                                           # BGR
Label = namedtuple('Label', ['name', 'id', 'color'])

labels = [
    Label('road'          ,     1,      (128,  64, 128)),
    Label('sidewalk'      ,     2,      (232,  35, 244)),
    Label('terrain'       ,     3,      (152, 251, 152)),
    Label('car'           ,     4,      (142,   0,   0)),
    Label('person'        ,     5,      (60 ,  20, 220)),
    Label('traffic sign'  ,     6,      (0  , 220, 220)),
    Label('traffic light' ,     7,      (30 , 170, 250)),
    Label('pole'          ,     8,      (153, 153, 153)),
    Label('parking'       ,     9,      (160, 170, 250)),
    Label('ground'        ,    10,      ( 81,   0,  81)),
]

names2labels = {label.name : label for label in labels}
