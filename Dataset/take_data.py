import csv

# Đọc dữ liệu từ file CSV gốc
with open('Clean_Dataset.csv', 'r') as f:
    reader = csv.reader(f)
    rows = list(reader)

first_1000_rows = rows[10000:]

# Ghi kết quả vào tệp CSV mới
with open('output.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(first_1000_rows)