import yaml

# Read YAML file
with open("config.yaml", 'r') as stream:
    try:
        yaml_data = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

print(yaml_data['checkpoints']) # prints ./ddns-api-key.yaml
print(type(yaml_data['checkpoints']))
for host in yaml_data['hosts']:
    print(type(host))
    print(host)