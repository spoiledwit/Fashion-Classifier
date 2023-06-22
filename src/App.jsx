import React, { useEffect, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import LinearProgress from '@mui/material/LinearProgress';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import { styled } from '@mui/system';
import Grid from '@mui/material/Grid';
import { CloudUploadOutlined } from '@mui/icons-material';
import { CircularProgress } from '@mui/material';

const AppContainer = styled('div')({
  background: 'linear-gradient(to bottom right, #303F9F, #C5CAE9)',
  minHeight: '100vh',
  padding: '20px',
  boxSizing: 'border-box',
  color: '#212121',
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  fontFamily: 'Roboto, sans-serif',
  textAlign: 'center'
});

const UploadButton = styled('label')({
  display: 'inline-flex',
  alignItems: 'center',
  justifyContent: 'center',
  padding: '10px 20px',
  background: '#3F51B5',
  color: 'white',
  borderRadius: '4px',
  cursor: 'pointer',
  marginTop: '20px',
  transition: 'background .3s',
  '&:hover': {
    background: '#303F9F'
  }
});

const ImageContainer = styled('div')({
  width: '100%',
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  padding: '10px',
  boxSizing: 'border-box',
  overflow: 'hidden',
  img: {
    maxWidth: '100%',
    borderRadius: '10px',
    boxShadow: '2px 2px 10px rgba(0, 0, 0, 0.3)'
  }
});

function App() {
  const model = useRef();
  const [prediction, setPrediction] = useState([]);
  const [errorMessage, setErrorMessage] = useState(null);
  const [originalImage, setOriginalImage] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
  'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'];

  useEffect(() => {
    async function loadModel() {
      try {
        model.current = await tf.loadLayersModel('model.json');
        console.log('Model loaded');
      } catch (err) {
        setErrorMessage('Failed to load model');
        console.error(err);
      }
    }
    loadModel();
  }, []);

  async function makePrediction(imageFile) {
    // Set the original image
    setOriginalImage(URL.createObjectURL(imageFile));
    try {
      // Create an Image object
      const img = new Image();
      img.src = URL.createObjectURL(imageFile);
      await img.decode(); // Wait for the image to be loaded

      // Create a tensor from the image
      const tensor = tf.browser.fromPixels(img, 1);
      const resized = tf.image.resizeBilinear(tensor, [28, 28]);
      const normalized = resized.div(tf.scalar(255.0));

      // Calculate the average pixel value of the image
      const avgPixelValue = normalized.mean().dataSync()[0];

      // Invert the color if the image is more white than black
      const possiblyInverted = avgPixelValue > 0.5 ? tf.scalar(1.0).sub(normalized) : normalized;

      // Add a dimension to match the input shape the model was trained with
      const input = possiblyInverted.reshape([1, 28, 28]);

      // Use the model for prediction
      const predictions = model.current.predict(input);

      // Set the prediction state
      const predictionData = predictions.arraySync()[0];
      const mappedPredictionData = predictionData.map((prediction, index) => {
        return { className: classes[index], prediction: prediction.toFixed(2) };
      });

      setPrediction(mappedPredictionData);

      // Convert the processed image back to an ImageData object
      const imageData = tf.browser.toPixels(possiblyInverted.squeeze()).then((data) => {
        const canvas = document.createElement('canvas');
        canvas.width = 28;
        canvas.height = 28;
        const ctx = canvas.getContext('2d');
        const imageData = new ImageData(data, 28, 28);
        ctx.putImageData(imageData, 0, 0);
        setProcessedImage(canvas.toDataURL());
      });
    } catch (err) {
      setErrorMessage('Failed to make prediction');
      console.error(err);
    }
  }  

  return (
    <AppContainer>
      <Typography variant="h2" gutterBottom>
        Fashion MNIST Classifier
      </Typography>
      <Typography variant="subtitle1" gutterBottom>
        Upload a clothing item among shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, and ankle boot
          and see what the model predicts it to be.
      </Typography>
      {errorMessage && <div>Error: {errorMessage}</div>}
      <UploadButton>
        <CloudUploadOutlined style={{ marginRight: '5px' }} />
        Upload an image
        <input type="file" accept="image/*" hidden onChange={e => makePrediction(e.target.files[0])} />
      </UploadButton>
      {originalImage && processedImage ? (
        <Grid container spacing={2} style={{ marginTop: '20px' }}>
          <Grid item xs={6}>
            <ImageContainer>
              <Typography variant="h6">Original Image</Typography>
              <img src={originalImage} alt="Original" />
            </ImageContainer>
          </Grid>
          <Grid item xs={6}>
            <ImageContainer>
              <Typography variant="h6">Processed Image</Typography>
              <img src={processedImage} alt="Processed" />
            </ImageContainer>
          </Grid>
        </Grid>
      ) : (
        <></>
      )}
      <Typography variant="h5" style={{ marginTop: '20px', width: '100%' }}>Predictions</Typography>
      {prediction.length > 0 && prediction.map((item, index) => (
        <Box key={index} sx={{ width: '100%', marginTop: 2 }}>
          <Typography variant="caption">{item.className}</Typography>
          <LinearProgress variant="determinate" value={Number(item.prediction)*100} />
        </Box>
      ))}
    </AppContainer>

  );
}

export default App;
