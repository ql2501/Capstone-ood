classname_dic = {
    "Imagenet_Openood" : {
        "classes":[
            'id',
            'ood',
        ],
        "templates":[
            'a photo of number: "{}".',
        ]
        },
    "mnist" : {
        "classes":[
            '0',
            '1',
            '2',
            '3',
            '4',
            '5',
            '6',
            '7',
            '8',
            '9',
        ],
        "templates":[
            'a photo of the number: "{}".',
        ]
        },
    "cifar10" : {
        "classes":[
            'airplane',
            'automobile',
            'bird',
            'cat',
            'deer',
            'dog',
            'frog',
            'horse',
            'ship',
            'truck',
        ],
        "templates":[
            'a photo of a {}.',
            'a blurry photo of a {}.',
            'a black and white photo of a {}.',
            'a low contrast photo of a {}.',
            'a high contrast photo of a {}.',
            'a bad photo of a {}.',
            'a good photo of a {}.',
            'a photo of a small {}.',
            'a photo of a big {}.',
            'a photo of the {}.',
            'a blurry photo of the {}.',
            'a black and white photo of the {}.',
            'a low contrast photo of the {}.',
            'a high contrast photo of the {}.',
            'a bad photo of the {}.',
            'a good photo of the {}.',
            'a photo of the small {}.',
            'a photo of the big {}.',
        ]
        },
    "cifar100" : {
        "classes":[
            'apple',
            'aquarium fish',
            'baby',
            'bear',
            'beaver',
            'bed',
            'bee',
            'beetle',
            'bicycle',
            'bottle',
            'bowl',
            'boy',
            'bridge',
            'bus',
            'butterfly',
            'camel',
            'can',
            'castle',
            'caterpillar',
            'cattle',
            'chair',
            'chimpanzee',
            'clock',
            'cloud',
            'cockroach',
            'couch',
            'crab',
            'crocodile',
            'cup',
            'dinosaur',
            'dolphin',
            'elephant',
            'flatfish',
            'forest',
            'fox',
            'girl',
            'hamster',
            'house',
            'kangaroo',
            'keyboard',
            'lamp',
            'lawn mower',
            'leopard',
            'lion',
            'lizard',
            'lobster',
            'man',
            'maple tree',
            'motorcycle',
            'mountain',
            'mouse',
            'mushroom',
            'oak tree',
            'orange',
            'orchid',
            'otter',
            'palm tree',
            'pear',
            'pickup truck',
            'pine tree',
            'plain',
            'plate',
            'poppy',
            'porcupine',
            'possum',
            'rabbit',
            'raccoon',
            'ray',
            'road',
            'rocket',
            'rose',
            'sea',
            'seal',
            'shark',
            'shrew',
            'skunk',
            'skyscraper',
            'snail',
            'snake',
            'spider',
            'squirrel',
            'streetcar',
            'sunflower',
            'sweet pepper',
            'table',
            'tank',
            'telephone',
            'television',
            'tiger',
            'tractor',
            'train',
            'trout',
            'tulip',
            'turtle',
            'wardrobe',
            'whale',
            'willow tree',
            'wolf',
            'woman',
            'worm',
        ],
        "templates":[
            'a photo of a {}.',
            'a blurry photo of a {}.',
            'a black and white photo of a {}.',
            'a low contrast photo of a {}.',
            'a high contrast photo of a {}.',
            'a bad photo of a {}.',
            'a good photo of a {}.',
            'a photo of a small {}.',
            'a photo of a big {}.',
            'a photo of the {}.',
            'a blurry photo of the {}.',
            'a black and white photo of the {}.',
            'a low contrast photo of the {}.',
            'a high contrast photo of the {}.',
            'a bad photo of the {}.',
            'a good photo of the {}.',
            'a photo of the small {}.',
            'a photo of the big {}.',
        ]
        },
    "ImageNet" : { # was tiny_imagenet, temporarily rename to ImageNet to unblock development
        "classes":[
            'goldfish',
            'European fire salamander',
            'bullfrog',
            'tailed frog',
            'American alligator',
            'boa constrictor',
            'trilobite',
            'scorpion',
            'black widow',
            'tarantula',
            'centipede',
            'goose',
            'koala',
            'jellyfish',
            'brain coral',
            'snail',
            'slug',
            'sea slug',
            'American lobster',
            'spiny lobster',
            'black stork',
            'king penguin',
            'albatross',
            'dugong',
            'Chihuahua',
            'Yorkshire terrier',
            'golden retriever',
            'Labrador retriever',
            'German shepherd',
            'standard poodle',
            'tabby',
            'Persian cat',
            'Egyptian cat',
            'cougar',
            'lion',
            'brown bear',
            'ladybug',
            'fly',
            'bee',
            'grasshopper',
            'walking stick',
            'cockroach',
            'mantis',
            'dragonfly',
            'monarch',
            'sulphur butterfly',
            'sea cucumber',
            'guinea pig',
            'hog',
            'ox',
            'bison',
            'bighorn',
            'gazelle',
            'Arabian camel',
            'orangutan',
            'chimpanzee',
            'baboon',
            'African elephant',
            'lesser panda',
            'abacus',
            'academic gown',
            'altar',
            'apron',
            'backpack',
            'bannister',
            'barbershop',
            'barn',
            'barrel',
            'basketball',
            'bathtub',
            'beach wagon',
            'beacon',
            'beaker',
            'beer bottle',
            'bikini',
            'binoculars',
            'birdhouse',
            'bow tie',
            'brass',
            'broom',
            'bucket',
            'bullet train',
            'butcher shop',
            'candle',
            'cannon',
            'cardigan',
            'cash machine',
            'CD player',
            'chain',
            'chest',
            'Christmas stocking',
            'cliff dwelling',
            'computer keyboard',
            'confectionery',
            'convertible',
            'crane',
            'dam',
            'desk',
            'dining table',
            'drumstick',
            'dumbbell',
            'flagpole',
            'fountain',
            'freight car',
            'frying pan',
            'fur coat',
            'gasmask',
            'go-kart',
            'gondola',
            'hourglass',
            'iPod',
            'jinrikisha',
            'kimono',
            'lampshade',
            'lawn mower',
            'lifeboat',
            'limousine',
            'magnetic compass',
            'maypole',
            'military uniform',
            'miniskirt',
            'moving van',
            'nail',
            'neck brace',
            'obelisk',
            'oboe',
            'organ',
            'parking meter',
            'pay-phone',
            'picket fence',
            'pill bottle',
            'plunger',
            'pole',
            'police van',
            'poncho',
            'pop bottle',
            "potter's wheel",
            'projectile',
            'punching bag',
            'reel',
            'refrigerator',
            'remote control',
            'rocking chair',
            'rugby ball',
            'sandal',
            'school bus',
            'scoreboard',
            'sewing machine',
            'snorkel',
            'sock',
            'sombrero',
            'space heater',
            'spider web',
            'sports car',
            'steel arch bridge',
            'stopwatch',
            'sunglasses',
            'suspension bridge',
            'swimming trunks',
            'syringe',
            'teapot',
            'teddy',
            'thatch',
            'torch',
            'tractor',
            'triumphal arch',
            'trolleybus',
            'turnstile',
            'umbrella',
            'vestment',
            'viaduct',
            'volleyball',
            'water jug',
            'water tower',
            'wok',
            'wooden spoon',
            'comic book',
            'plate',
            'guacamole',
            'ice cream',
            'ice lolly',
            'pretzel',
            'mashed potato',
            'cauliflower',
            'bell pepper',
            'mushroom',
            'orange',
            'lemon',
            'banana',
            'pomegranate',
            'meat loaf',
            'pizza',
            'potpie',
            'espresso',
            'alp',
            'cliff',
            'coral reef',
            'lakeside',
            'seashore',
            'acorn',
        ],
        "templates":[
            'a photo of a {}.',
            'a blurry photo of a {}.',
            'a black and white photo of a {}.',
            'a low contrast photo of a {}.',
            'a high contrast photo of a {}.',
            'a bad photo of a {}.',
            'a good photo of a {}.',
            'a photo of a small {}.',
            'a photo of a big {}.',
            'a photo of the {}.',
            'a blurry photo of the {}.',
            'a black and white photo of the {}.',
            'a low contrast photo of the {}.',
            'a high contrast photo of the {}.',
            'a bad photo of the {}.',
            'a good photo of the {}.',
            'a photo of the small {}.',
            'a photo of the big {}.',
        ]
        }
}