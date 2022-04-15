import React, { useState, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";

import {
  Group,
  Text,
  useMantineTheme,
  MantineTheme,
  Stack,
  Chip,
  Title,
  LoadingOverlay,
} from "@mantine/core";
import { Upload, Photo, X, Icon as TablerIcon } from "tabler-icons-react";
import { Dropzone, DropzoneStatus, IMAGE_MIME_TYPE } from "@mantine/dropzone";

function getIconColor(status, theme) {
  return status.accepted
    ? theme.colors[theme.primaryColor][theme.colorScheme === "dark" ? 4 : 6]
    : status.rejected
    ? theme.colors.red[theme.colorScheme === "dark" ? 4 : 6]
    : theme.colorScheme === "dark"
    ? theme.colors.dark[0]
    : theme.colors.gray[7];
}

function ImageUploadIcon({ status, ...props }) {
  if (status.accepted) {
    return <Upload {...props} />;
  }

  if (status.rejected) {
    return <X {...props} />;
  }

  return <Photo {...props} />;
}

export const dropzoneChildren = (status, theme) => (
  <Group
    position="center"
    spacing="xl"
    style={{ minHeight: 220, pointerEvents: "none" }}
  >
    <ImageUploadIcon
      status={status}
      style={{ color: getIconColor(status, theme) }}
      size={80}
    />

    <div>
      <Text size="xl" inline>
        Drag images here or click to select files
      </Text>
      <Text size="sm" color="dimmed" inline mt={7}>
        Attach as many files as you like, each file should not exceed 5mb
      </Text>
    </div>
  </Group>
);

const readImageFile = (file) => {
  return new Promise((resolve) => {
    const reader = new FileReader();

    reader.onload = () => resolve(reader.result);

    reader.readAsDataURL(file);
  });
};

const createHTMLImageElement = (imageSrc) => {
  return new Promise((resolve) => {
    const img = new Image();

    img.onload = () => resolve(img);

    img.src = imageSrc;
  });
};

function App() {
  const theme = useMantineTheme();

  const [loading, setLoading] = useState(false);
  const [confidence, setConfidence] = useState(null);
  const [predictedClass, setPredictedClass] = useState(null);

  const [model, setModel] = useState(null);
  const [classLabels, setClassLabels] = useState(null);

  const handleImageChange = async (files) => {
    if (files.length === 0) {
      setConfidence(null);
      setPredictedClass(null);
    }

    if (files.length === 1) {
      setLoading(true);

      const imageSrc = await readImageFile(files[0]);
      const image = await createHTMLImageElement(imageSrc);

      // tf.tidy for automatic memory cleanup
      const [predictedClass, confidence] = tf.tidy(() => {
        const tensorImg = tf.browser
          .fromPixels(image)
          .resizeNearestNeighbor([256, 256])
          .toFloat()
          .expandDims();
        const result = model.predict(tensorImg);

        const predictions = result.dataSync();
        const predicted_index = result.as1D().argMax().dataSync()[0];
        console.log("predictions");
        console.log(predictions);
        console.log("predicted_index");
        console.log(predicted_index);

        const predictedClass = classLabels[predicted_index];
        const confidence = Math.round(predictions[predicted_index] * 100);

        return [predictedClass, confidence];
      });

      setPredictedClass(predictedClass);
      setConfidence(confidence);
      setLoading(false);
    }
  };

  useEffect(() => {
    const loadModel = async () => {
      const model_url = "model/model.json";

      const model = await tf.loadGraphModel(model_url);

      setModel(model);
    };

    const getClassLabels = async () => {
      const data = ["canada_ontario_cvo", "none"];

      setClassLabels(data);
    };

    loadModel();
    getClassLabels();
  }, []);

  return (
    <div className="App">
      <Title order={1}>
        CSI4106 Project: Veterinary Certificate of Accreditation Classifier
      </Title>
      <Title order={3}>Firas Jribi</Title>
      <Title order={3}>Emilie Fortin</Title>
      <Title order={3}>Abir Boutahri</Title>
      <LoadingOverlay visible={loading} />
      <Dropzone
        onDrop={(files) => {
          console.log("accepted files", files);
          handleImageChange(files);
        }}
        onReject={(files) => console.log("rejected files", files)}
        maxSize={3 * 1024 ** 2}
        accept={IMAGE_MIME_TYPE}
      >
        {(status) => dropzoneChildren(status, theme)}
      </Dropzone>
      <Stack
        style={{ marginTop: "2em", width: "12rem" }}
        direction="row"
        spacing={1}
      >
        <Title order={4}>
          {predictedClass === null
            ? "Prediction:"
            : `Prediction: ${predictedClass}`}
        </Title>
        <Title order={4}>
          {confidence === null ? "Confidence:" : `Confidence: ${confidence}%`}
        </Title>
      </Stack>
    </div>
  );
}

export default App;
