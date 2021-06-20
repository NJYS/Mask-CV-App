import React, {useRef, useCallback} from 'react';
import Webcam from 'react-webcam';
import { Button, Grid, Box } from '@material-ui/core';

const videoConstraints = {
    width: 448,
    height: 448,
    facingMode: "user"
};

interface prop {
  camToggle  : () => void,
  setPreview : React.Dispatch<React.SetStateAction<string>>
}

const WebcamCapture = (props : prop) => {
    const webcamRef = useRef<any>(null);
    const {camToggle, setPreview} = props;

    const capture = useCallback(() => {
        const imageSrc = webcamRef.current.getScreenshot();
        setPreview(imageSrc) // base64 image file
        camToggle();
      }, [webcamRef, camToggle, setPreview]);

    return (
          <>
            <Grid container justify="center" alignItems="center">
              <Webcam
                audio={false}
                height={window.innerHeight> 800 ? 448 : 224}
                ref={webcamRef}
                screenshotFormat="image/jpeg"
                width={window.innerWidth > 800 ? 448 : 224}
                videoConstraints={videoConstraints}
              />
            </Grid>
            <Box mt = "1rem"/>
            <Grid container spacing={5} direction="row" alignItems="center" justify="center">
              <Grid item>
                <Button variant="contained" size="small" color="primary" onClick={capture}>캡쳐</Button>
              </Grid>
              <Grid item>
                <Button variant="contained" size="small" color="primary" onClick={camToggle}>종료</Button>
              </Grid>
            </Grid>
        </>
    );
}

export default WebcamCapture
