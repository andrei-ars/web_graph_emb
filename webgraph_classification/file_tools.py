import os


def get_html_file_paths_and_labels(input_dir):

    labels = []
    file_paths = []

    for label in os.listdir(input_dir):
        
        subdir_path = os.path.join(input_dir, label)
        filenames = os.listdir(subdir_path)
        for filename in filenames:
            file_path = os.path.join(subdir_path, filename)
            file_paths.append(file_path)
            #html_files.append(filename)
            labels.append(label)

    return file_paths, labels


if __name__ == "__main__":

    file_paths, labels = get_html_file_paths_and_labels(input_dir="dataset/html")
    print("labels:", labels)
    print("file_paths:", file_paths)