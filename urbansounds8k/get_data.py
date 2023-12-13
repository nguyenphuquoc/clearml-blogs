import pandas as pd
from pathlib import Path
from clearml import Task, Dataset, StorageManager

task = Task.init(project_name='examples/Urbansounds',
                 task_name='download data')

configuration = {
    'selected_classes': ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling',
                         'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']
}
task.connect(configuration)


def get_urbansound8k():
    # Tải xuống bộ dữ liệu UrbanSound8K (https://urbansounddataset.weebly.com/urbansound8k.html)
    # Để đơn giản, sử dụng ở đây một tập hợp con của tập dữ liệu đó bằng Clearml StorageManager
    path_to_urbansound8k = StorageManager.get_local_copy(
        "https://allegro-datasets.s3.amazonaws.com/clearml/UrbanSound8K.zip",
        extract_archive=True)
    path_to_urbansound8k_csv = Path(path_to_urbansound8k) / 'UrbanSound8K' / 'metadata' / 'UrbanSound8K.csv'
    path_to_urbansound8k_audio = Path(path_to_urbansound8k) / 'UrbanSound8K' / 'audio'

    return path_to_urbansound8k_csv, path_to_urbansound8k_audio


def log_dataset_statistics(dataset, metadata):
    histogram_data = metadata['class'].value_counts()
    dataset.get_logger().report_table(
        title='Raw Dataset Metadata',
        series='Raw Dataset Metadata',
        table_plot=metadata
    )
    dataset.get_logger().report_histogram(
        title='Class distribution',
        series='Class distribution',
        values=histogram_data,
        iteration=0,
        xlabels=histogram_data.index.tolist(),
        yaxis='Amount of samples'
    )


def build_clearml_dataset():
    # Nhận bản sao cục bộ của cả dữ liệu và nhãn
    path_to_urbansound8k_csv, path_to_urbansound8k_audio = get_urbansound8k()
    urbansound8k_metadata = pd.read_csv(path_to_urbansound8k_csv)
    # Tập hợp dữ liệu để chỉ bao gồm các lớp mong muốn
    urbansound8k_metadata = \
        urbansound8k_metadata[urbansound8k_metadata['class'].isin(configuration['selected_classes'])]

    # Tạo khung dữ liệu gấu trúc chứa nhãn và thông tin khác mà chúng tôi cần sau này (gấp lại để phân chia thử nghiệm tàu)
    metadata = pd.DataFrame({
        'fold': urbansound8k_metadata.loc[:, 'fold'],
        'filepath': ('fold' + urbansound8k_metadata.loc[:, 'fold'].astype(str)
                     + '/' + urbansound8k_metadata.loc[:, 'slice_file_name'].astype(str)),
        'label': urbansound8k_metadata.loc[:, 'classID']
    })

    # Bây giờ, hãy tạo tập dữ liệu Clearml để bắt đầu lập phiên bản các thay đổi và giúp việc lấy dữ liệu phù hợp dễ dàng hơn nhiều
    # trong các tác vụ khác cũng như trên các máy khác nhau
    dataset = Dataset.create(
        dataset_name='UrbanSounds example',
        dataset_project='examples/Urbansounds',
        dataset_tags=['raw']
    )

    # Thêm các tệp cục bộ đã tải xuống trước đó
    dataset.add_files(path_to_urbansound8k_audio)
    # Thêm siêu dữ liệu ở định dạng gấu trúc, giờ đây có thể thấy siêu dữ liệu đó trong webUI và có thể truy cập dễ dàng
    dataset._task.upload_artifact(name='metadata', artifact_object=metadata)
    # Hãy thêm một số biểu đồ thú vị làm số liệu thống kê trong phần lô!
    log_dataset_statistics(dataset, urbansound8k_metadata)
    # Hoàn thiện và tải lên dữ liệu và nhãn của tập dữ liệu
    dataset.finalize(auto_upload=True)


if __name__ == '__main__':
    build_clearml_dataset()
