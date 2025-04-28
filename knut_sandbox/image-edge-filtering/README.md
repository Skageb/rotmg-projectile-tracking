# Image Edge Filtering Project

This project applies various edge filters to an image using Python. The image is loaded from the `testing_folder`, and different filters are applied with their processing times recorded.

## Project Structure

```
image-edge-filtering
├── src
│   ├── main.py         # Entry point of the application
│   └── filters.py      # Contains edge filter functions
├── testing_folder       # Contains the image file "test_image.png"
├── requirements.txt     # Lists project dependencies
└── README.md            # Project documentation
```

## Requirements

To run this project, you need to install the required dependencies. You can do this by running:

```
pip install -r requirements.txt
```

## Usage

1. Place your image file named `test_image.png` in the `testing_folder`.
2. Run the application by executing the following command in your terminal:

```
python src/main.py
```

3. The application will apply various edge filters to the image and print the processing times for each filter.

## Filters

The project includes several edge filters implemented in `filters.py`. Each filter function takes the image and specific parameters to customize the filtering process.

## Contributing

Feel free to fork the repository and submit pull requests for any improvements or additional features.