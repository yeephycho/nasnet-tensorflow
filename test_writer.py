import json
# content = [['prob', 0.999],['label', 'bankcard'],['filepath', '1.jpg']]
content = {}
res = []
content['prob'] = 0.999
content['label'] = 'bankcard'
content['filepath'] = '1.jpg'

content2 = {}
content2['prob'] = 0.999
content2['label'] = 'bankcard'
content2['filepath'] = '1.jpg'

res.append(content)
res.append(content2)

print(res)

processed_files = ['1.tfrecord', '2.tfrecord']

model = 'nasnet_shangshu'
'''
with open('./model_output/processed_files/test_{}_processed_files.json'.format(model), 'w') as f:
    f.write(json.dump(processed_files,fp))
'''
with open('./model_output/classify_result/test_{}_classify_result.json'.format(model), 'w') as f:
    f.write(json.dumps(content, indent=4, separators=(',',':')))
