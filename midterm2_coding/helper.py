from tensorflow.keras.preprocessing import image
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import zipfile
import tempfile


def plot_df(df):
    # Replace integer values with category names
    names_ethnicity = ['white', 'black', 'asian', 'indian', 'other']
    name_genders = ['male', 'female']
    df2 = df.copy()
    df2['ethnicity'] = df['ethnicity'].apply(lambda x: names_ethnicity[x])
    df2['gender'] = df['gender'].apply(lambda x: name_genders[x])

    # Set the style of the plot
    sns.set(style="whitegrid")

    # Create a figure with three subplots
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))

    # Plot age distribution
    sns.histplot(data=df, x='age', ax=axes[0], color='blue', kde=True)
    axes[0].set_title('Age Distribution')

    # Plot gender representation
    sns.countplot(data=df2, x='gender', ax=axes[1], hue='gender', palette='coolwarm')
    axes[1].set_title('Gender Representation')

    # Plot ethnicity representation
    sns.countplot(data=df2, x='ethnicity', ax=axes[2], hue='ethnicity', palette='viridis')
    axes[2].set_title('Ethnicity Representation')
    plt.tight_layout()



def resample(df, unbalanced_col='ethnicity', n_per_class=1500):

    unique_classes = df[unbalanced_col].unique()
    dfs = []
    for c in unique_classes:
        replace = False
        to_sample = n_per_class
        cur_df = df[df[unbalanced_col] == c]
        if cur_df.shape[0] < n_per_class:
            replace = True
            to_sample = n_per_class - cur_df.shape[0]
        sample = cur_df.sample(to_sample, replace=replace)
        if cur_df.shape[0] < n_per_class:
            sample = pd.concat([cur_df, sample], axis=0)
        dfs.append(sample)
    df_resampled = pd.concat(dfs).sample(frac=1).reset_index(drop=True)
    
    return df_resampled


# Function to load and preprocess an image
def load_and_preprocess_image(img_path, target_size=(28,28)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    # img_array = preprocess_input(img_array)
    return img_array

# Generator function
def data_generator(df, base_img_path, img_size=(28,28), batch_size=16):
    while True:
        # Shuffle the dataframe
        df = df.sample(frac=1).reset_index(drop=True)
        
        # Create batches
        for i in range(0, len(df), batch_size):
            batch_df = df[i:i+batch_size]
            
            batch_imgs = []
            batch_ages = []
            batch_eths = []
            batch_gens = []

            for index, row in batch_df.iterrows():
                img_path = os.path.join(base_img_path, row['img_name'])
                img = load_and_preprocess_image(img_path, target_size=img_size)
                age = row['age']
                eth = row['ethnicity']
                gen = row['gender']

                batch_imgs.append(img)
                batch_ages.append(age)
                batch_eths.append(eth)
                batch_gens.append(gen)

            yield (tf.stack(batch_imgs), 
                   (tf.expand_dims(batch_ages, -1), tf.cast(batch_eths, tf.int32), tf.expand_dims(batch_gens, -1)))


def data_generator2(df, image_data, batch_size=16):
    while True:
        # Shuffle the dataframe
        df = df.sample(frac=1).reset_index(drop=True)
        
        # Create batches
        for i in range(0, len(df), batch_size):
            batch_df = df[i:i+batch_size]
            
            batch_imgs = []
            batch_ages = []
            batch_eths = []
            batch_gens = []

            for index, row in batch_df.iterrows():
                img = image_data[index]
                img = img / 255.0  # Normalize pixel values
                img = np.expand_dims(img, axis=-1)  # Add channel dimension
                age = row['age']
                eth = row['ethnicity']
                gen = row['gender']

                batch_imgs.append(img)
                batch_ages.append(age)
                batch_eths.append(eth)
                batch_gens.append(gen)

            yield (tf.stack(batch_imgs), 
                   (tf.expand_dims(batch_ages, -1), tf.cast(batch_eths, tf.int32), tf.expand_dims(batch_gens, -1)))

def decode_image(image_data, img_dim):
    image = tf.image.decode_png(image_data, channels=1)
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    image = tf.image.grayscale_to_rgb(image)
    image = tf.image.resize(image, [img_dim[0], img_dim[1]])
    image.set_shape((*img_dim, 3))
    return image

def parse_example(example_proto, img_dim):
    feature_description = {
        'age': tf.io.FixedLenFeature([], tf.float32),
        'ethnicity': tf.io.FixedLenFeature([], tf.int64),
        'gender': tf.io.FixedLenFeature([], tf.float32),
        'img': tf.io.FixedLenFeature([], tf.string),
    }

    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    img = decode_image(parsed_features['img'], img_dim)

    return img, (parsed_features['age'], parsed_features['ethnicity'], parsed_features['gender'])

def load_tfrecord_dataset(tfrecord_file, img_dim, batch_size):
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    dataset = dataset.map(lambda x: parse_example(x, img_dim), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def load_compressed_model(filepath):
    # Unzip the compressed model file to a temporary HDF5 file
    temp_h5_file = tempfile.NamedTemporaryFile(delete=True, suffix='.h5')
    temp_h5_file_name = temp_h5_file.name

    with zipfile.ZipFile(filepath, mode='r') as zf:
        for name in zf.namelist():
            if name.endswith('.h5'):
                with open(temp_h5_file_name, mode='wb') as f:
                    f.write(zf.read(name))
                break

    # Load the model from the temporary HDF5 file
    model = tf.keras.models.load_model(temp_h5_file_name)

    # Close and delete the temporary HDF5 file
    temp_h5_file.close()

    return model
