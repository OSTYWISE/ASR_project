from torch.nn.utils.rnn import pad_sequence
import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    # Commented code is for num_channels > 1
    audios = [item['audio'].squeeze() for item in dataset_items]
    # audios = [item['audio'].transpose(0, 1) for item in dataset_items]
    spectrograms = [item['spectrogram'].squeeze().transpose(0, 1) for item in dataset_items]
    # spectrograms = [item['spectrogram'].transpose(0, 2) for item in dataset_items]
    spectrogram_lengths = torch.Tensor([item['spectrogram'].shape[2] for item in dataset_items]).to(torch.int32)
    texts = [item['text'] for item in dataset_items]
    text_encoded = [item['text_encoded'].squeeze() for item in dataset_items]
    # text_encoded = [item['text_encoded'].transpose(0, 1) for item in dataset_items]
    text_encoded_length = torch.Tensor([item['text_encoded'].shape[1] for item in dataset_items]).to(torch.int32)
    audio_paths = [item['audio_path'] for item in dataset_items]

    # Pad audios, spectrograms and text_encoded sequences
    padded_audios = pad_sequence(audios, batch_first=True, padding_value=0)  # .transpose(1, 2)
    padded_spectrograms = pad_sequence(spectrograms, batch_first=True, padding_value=0).transpose(1, 2)  # .transpose(1, 3)
    padded_text_encoded = pad_sequence(text_encoded, batch_first=True, padding_value=0)  # .transpose(1, 2)

    # Create the result batch dictionary
    result_batch = {
        'audio': padded_audios,
        'spectrogram': padded_spectrograms,
        'text_encoded': padded_text_encoded,
        'text': texts,
        'audio_path': audio_paths,
        'spectrogram_length': spectrogram_lengths,
        'text_encoded_length': text_encoded_length
    }
    return result_batch
