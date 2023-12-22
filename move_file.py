import os
import shutil

def create_directories(source_dir, dest_dir):
    files = os.listdir(source_dir)
    file_count = 0
    dir_count = 0
    dest_subdir = os.path.join(dest_dir, "c" + str(dir_count))
    os.makedirs(dest_subdir, exist_ok=True)

    for file_name in sorted(files):
        print(file_name, ' ', file_count)
        source_file = os.path.join(source_dir, file_name)
        dest_file = os.path.join(dest_subdir, file_name)
        file_count += 1

        if file_count == 5:
            dir_count += 1
            dest_subdir = os.path.join(dest_dir, "c" + str(dir_count))
            os.makedirs(dest_subdir, exist_ok=True)
            file_count = 0
            print(".....")

        shutil.move(source_file, dest_file)


    print("Directories created successfully!")

# Specify the source directory and destination directory for the files

for i in range(9):
    i = i + 1
    source_directory = "./sample"+str(i)+"/"
    destination_directory = "./t"+str(i)+"/"

    create_directories(source_directory, destination_directory)
