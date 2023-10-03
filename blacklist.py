file = open('text.txt', 'w')
for key, value in a.items():
  file.write(f'{key}, {value}\n')
file.close()

self.record_file.write(f'{str(packet._asdict())}\n')
