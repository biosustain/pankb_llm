from pymongo import MongoClient

client = MongoClient(
      host='localhost',
      port=27017,
      username='pankbDbOwner',
      password='t7rgfbsajdbSA',
      authSource='pankb',
      authMechanism='SCRAM-SHA-1'
  )

db = client['pankb']
col = db['pankb_publications']


col.insert_one({
      'title':'Comparative Genomic Assessment of the Cupriavidus necator Species for One-Carbon Based Biomanufacturing',
      'source':'https://doi.org/10.1111/1751-7915.70201'
    })

