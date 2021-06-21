import React, { useState, useEffect} from 'react';
import axios from 'axios';
import { useMutation } from 'react-query';
import Image from 'material-ui-image'

// components
import ImageResizer from './components/ImageResizer';
import WebcamCapture from './components/WebcamCapture';

// material UI
import { Box, Button, Grid, Container, CircularProgress, Tooltip, Typography, Paper } from '@material-ui/core';
import { makeStyles, createMuiTheme, ThemeProvider } from '@material-ui/core/styles';
import IconButton from '@material-ui/core/IconButton';
import PhotoCamera from '@material-ui/icons/PhotoCamera';
import AttachmentIcon from '@material-ui/icons/Attachment';
interface picture {
    image : string;
} 

const useStyles = makeStyles((theme) => ({
    root : {
      width : 500,
    },
    button: {
      '& > *': {
        margin: theme.spacing(1),
      },
    },
    input: {
      display: 'none',
    },
    main : {
        padding : '2em',
        backgroundColor: '#757de8',
        height: '60vh', 
        overflow: 'auto',
    },
    progressCircle : {
      margin:'1em'
    }
}));

//font theme
const theme = createMuiTheme();
theme.typography.h3 = {
  fontSize: '1.2rem',
  '@media (min-width:600px)': {
    fontSize: '1.5rem',
  },
  [theme.breakpoints.up('md')]: {
    fontSize: '2rem',
  },
};

function Home(){
    const classes = useStyles();
    const [previewURL, setPreview] = useState<string>('');
    const [isSetImage, setImage] = useState<boolean>(false);
    const [result, setResult] = useState<string>('');
    const [camState, setCam] = useState<boolean>(false);
    const [fileValue, setFile] = useState<HTMLInputElement>();

    const token = `${'njys'}:${'1q2w3e4r!'}`;
    const encodedToken = Buffer.from(token).toString('base64');
    const headers = { 'Authorization': 'Basic '+ encodedToken };
    const api = axios.create({
      baseURL: `https://boostcamp-nyjs.herokuapp.com/`,
      headers: headers
    })
  
    const [mutateCreate, {isLoading : isPicLoading}] = useMutation(
        (data: picture) => api.post('masks/', data), { 
      onSuccess: (res) => {
        setResult(res.data.result);
      },
      onError : () => {
        setResult('전송 오류');
      }
    })
    
    useEffect(() => {
      if(camState) {
          setPreview('');
          setImage(false);
        }
      else if(previewURL !== '') setImage(true);
    }, [camState, previewURL, setPreview, setImage]); 

    const Submit = (e : React.MouseEvent<HTMLButtonElement, MouseEvent>) => {
        e.preventDefault();
    
        if(previewURL === '') { 
          setResult('파일이 없습니다.');
          return;
        } 
    
        const json : string = JSON.stringify({
          image : previewURL,
        })
        const obj : picture = JSON.parse(json);
        mutateCreate(obj);
    }
    
    const handleFileInput = async (e : React.FormEvent<HTMLInputElement>) => {
      e.preventDefault();
      const target = e.target as HTMLInputElement;
      // 파일 읽기
      setFile(target);
      if(camState) setCam(false);

      try {
        if(target){
          let file : File = target.files![0];
          const image : string = await ImageResizer(file);
          setResult('');
          setPreview(image);
          setImage(true);
        }
      } catch { 
        // 오류 메시지 등록 후 박스 내 이미지 제거
        setResult('업로드 오류');
        setPreview('');
        setImage(false);
      }
    }

    const Loading = () => { 
      return ( 
        <Box display="flex" alignItems="center" justifyContent="center" className = {classes.progressCircle}>
          <CircularProgress color="primary" />
        </Box>      
      ) 
    }

    const Result = () => {
      return (
        <ThemeProvider theme={theme}>
          <br/>
          <Box display="flex" alignItems="center" justifyContent="center"><Typography variant="h3" id="res">{result}</Typography></Box>
        </ThemeProvider>
      )
    }

    // webcam
    const camToggle = () => {
      setResult('');
      setCam(camState => !camState);
      // 웹캠 켰을 때 기존 사진 정보 제거
      if(fileValue !== undefined) {
        fileValue.value = '';
      }
    }
  
    return (
      <>
      <Grid container spacing={0} direction="column" alignItems="center" justify="center">
          <Box mt="10rem"/>
        <Container maxWidth="sm">
          <Grid item>
          <Paper elevation={4} className ={classes.main}>
            {camState ? <WebcamCapture setPreview = {setPreview} camToggle ={camToggle}/> : null}
            {isSetImage?  <Image src={previewURL}/> : null}
            <Box mt="3rem"/>
            {isPicLoading ? <Loading/> : <Result/> }
          </Paper>
          </Grid>
        </Container>
        <Box mb="0.5rem"/>
        <Grid container spacing={5} direction="row" alignItems="center" justify="center">
          <Grid item>
            <label htmlFor="image_uploads">
            <Tooltip title="image upload" placement="left" arrow>
            <IconButton color="primary" aria-label="upload" component="span">
              <AttachmentIcon fontSize = 'large'/>
            </IconButton>
            </Tooltip>
            <input 
              type = "file" 
              id ="image_uploads" 
              accept="image/*" 
              onClick ={()=>{setResult(''); setPreview('');}}
              onChange={handleFileInput} 
              className = {classes.input} />
            </label>
          </Grid>
          <Grid item>
            <label htmlFor="open_webcam">
            <Tooltip title="webcam on" arrow>
            <IconButton color="primary" aria-label="open_webcam" component="span" onClick={camToggle} >
              <PhotoCamera fontSize = 'large'/>
            </IconButton>
            </Tooltip>
            </label>
          </Grid>
          <Grid item>
            <Button variant="contained" size="small" color="primary" onClick={Submit}>✨제출</Button>
          </Grid>
        </Grid>
      </Grid>
      <Box mb="0.5rem"/>
      </> 
    );
  }

  export default Home;