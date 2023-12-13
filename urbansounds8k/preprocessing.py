    import os.path
    from pathlib import Path

    import matplotlib as mpl
    import numpy as np
    from tqdm import tqdm
    import torchaudio
    import torch
    from clearml import Task, Dataset

    task = Task.init(project_name='examples/Urbansounds',
                     task_name='preprocessing')

    # Xử lý dữ liệu thô và tạo một tập dữ liệu mới đẩy lên ClearML
    # Dễ dàng gỡ lỗi và kiểm tra thủ công bằng cách thêm các mẫu gỡ lỗi vào tập dữ liệu
    # âm thanh gốc và biểu đồ phổ mel được xử lý của nó dưới dạng mẫu gỡ lỗi, vì vậy có thể kiểm tra thủ công
    # nếu mọi việc diễn ra theo đúng kế hoạch.


    class PreProcessor:
        def __init__(self):
            self.configuration = {
                'number_of_mel_filters': 64,
                'resample_freq': 22050
            }
            task.connect(self.configuration)

        def preprocess_sample(self, sample, original_sample_freq):
            if self.configuration['resample_freq'] > 0:
                resample_transform = torchaudio.transforms.Resample(orig_freq=original_sample_freq,
                                                                    new_freq=self.configuration['resample_freq'])
                sample = resample_transform(sample)

            sample = torch.mean(sample, dim=0, keepdim=True)
            melspectrogram_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.configuration['resample_freq'],
                n_mels=self.configuration['number_of_mel_filters']
            )
            melspectrogram = melspectrogram_transform(sample)
            melspectogram_db = torchaudio.transforms.AmplitudeToDB()(melspectrogram)
            fixed_length = 3 * (self.configuration['resample_freq'] // 200)
            if melspectogram_db.shape[2] < fixed_length:
                melspectogram_db = torch.nn.functional.pad(melspectogram_db, (0, fixed_length - melspectogram_db.shape[2]))
            else:
                melspectogram_db = melspectogram_db[:, :, :fixed_length]

            return melspectogram_db


    class DataSetBuilder:
        def __init__(self):
            self.configuration = {
                'dataset_path': 'dataset'
            }
            task.connect(self.configuration)

            self.original_dataset = Dataset.get(
                dataset_project='examples/Urbansounds',
                dataset_name='UrbanSounds example',
                dataset_tags=['raw'],
                alias='Raw Dataset'
            )
            # Điều này sẽ trả về khung dữ liệu gấu trúc đã thêm trong nhiệm vụ trước
            self.metadata = Task.get_task(task_id=self.original_dataset._task.id).artifacts['metadata'].get()
            # Điều này sẽ tải xuống dữ liệu và trả về đường dẫn cục bộ cho dữ liệu
            self.original_dataset_path = \
                Path(self.original_dataset.get_mutable_local_copy(self.configuration['dataset_path'], overwrite=True))

            # Chuẩn bị một bộ tiền xử lý sẽ xử lý từng mẫu một
            self.preprocessor = PreProcessor()

            # Hãy sẵn sàng cho cái mới
            self.preprocessed_dataset = None

        def log_dataset_statistics(self):
            histogram_data = self.metadata['label'].value_counts()
            self.preprocessed_dataset.get_logger().report_table(
                title='Raw Dataset Metadata',
                series='Raw Dataset Metadata',
                table_plot=self.metadata
            )
            self.preprocessed_dataset.get_logger().report_histogram(
                title='Class distribution',
                series='Class distribution',
                values=histogram_data,
                iteration=0,
                xlabels=histogram_data.index.tolist(),
                yaxis='Amount of samples'
            )

        def build_dataset(self):
            # Hãy tạo một tập dữ liệu mới là con của tập dữ liệu gốc
            # Thêm các mẫu được xử lý trước vào tập dữ liệu gốc để tạo ra phiên bản mới
            # Việc cung cấp tập dữ liệu gốc cho phép giữ được dòng dữ liệu rõ ràng
            self.preprocessed_dataset = Dataset.create(
                dataset_name='UrbanSounds example',
                dataset_project='examples/Urbansounds',
                dataset_tags=["preprocessed"],
                parent_datasets=[self.original_dataset.id]
            )

            # lặp qua các mục siêu dữ liệu và xử lý trước từng mẫu, sau đó thêm một số mẫu làm mẫu gỡ lỗi vào
            # Kiểm tra kỹ giao diện người dùng theo cách thủ công xem mọi thứ đã hoạt động chưa (có thể xem biểu đồ phổ và nghe
            # âm thanh cạnh nhau trong giao diện người dùng mẫu gỡ lỗi)
            for i, (_, data) in tqdm(enumerate(self.metadata.iterrows())):
                _, audio_file_path, label = data.tolist()
                sample, sample_freq = torchaudio.load(self.original_dataset_path / audio_file_path, normalize=True)
                spectrogram = self.preprocessor.preprocess_sample(sample, sample_freq)
                # Chỉ lấy tên tệp và thay thế phần mở rộng, lưu hình ảnh ở đây
                new_file_name = os.path.basename(audio_file_path).replace('.wav', '.npy')
                # Nhận đúng thư mục, về cơ bản là thư mục tập dữ liệu gốc + tên tệp mới
                spectrogram_path = self.original_dataset_path / os.path.dirname(audio_file_path) / new_file_name
                # Lưu mảng numpy vào đĩa
                np.save(spectrogram_path, spectrogram)

                # Ghi nhật ký mỗi mẫu thứ 10 dưới dạng mẫu gỡ lỗi vào giao diện người dùng để có thể kiểm tra thủ công
                if i % 10 == 0:
                    # Chuyển đổi mảng numpy thành JPEG có thể xem được
                    rgb_image = mpl.colormaps['viridis'](spectrogram[0, :, :].detach().numpy() * 255)[:, :, :3]
                    title = os.path.splitext(os.path.basename(audio_file_path))[0]

                    # Báo cáo hình ảnh và âm thanh gốc để có thể xem cạnh nhau
                    self.preprocessed_dataset.get_logger().report_image(
                        title=title,
                        series='spectrogram',
                        image=rgb_image
                    )
                    self.preprocessed_dataset.get_logger().report_media(
                        title=title,
                        series='original_audio',
                        local_path=self.original_dataset_path / audio_file_path
                    )
            # Đường dẫn dữ liệu ban đầu bây giờ cũng sẽ có các biểu đồ phổ trong cây tệp của nó.
            # Vì vậy, đó là lý do tại sao thêm nó vào đây để điền vào tập dữ liệu mới.
            self.preprocessed_dataset.add_files(self.original_dataset_path)
            # Một lần nữa thêm một số hình ảnh trực quan vào nhiệm vụ
            self.log_dataset_statistics()
            # đẩy lại siêu dữ liệu
            self.preprocessed_dataset._task.upload_artifact(name='metadata', artifact_object=self.metadata)
            self.preprocessed_dataset.finalize(auto_upload=True)


    if __name__ == '__main__':
        datasetbuilder = DataSetBuilder()
        datasetbuilder.build_dataset()
