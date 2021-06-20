import {useState} from 'react';
import ContactusTable from './ContactusTable';

// material-UI
import Box from '@material-ui/core/Box';
import { makeStyles } from '@material-ui/core/styles';
import { AppBar, Toolbar, IconButton, Typography, Modal, Backdrop, Fade, Tooltip } from '@material-ui/core';

import Brightness3Icon from "@material-ui/icons/Brightness3";
import Brightness7Icon from "@material-ui/icons/Brightness7";
import MailIcon from '@material-ui/icons/Mail';
import GitHubIcon from '@material-ui/icons/GitHub';

const useStyles = makeStyles((theme) => ({
  grow: {
    flexGrow: 1,
  },
  menuButton: {
    marginRight: theme.spacing(2),
  },
  title: {
    display: 'block'
  },
  inputRoot: {
    color: 'inherit',
  },
  inputInput: {
    padding: theme.spacing(1, 1, 1, 0),
    // vertical padding + font size from searchIcon
    paddingLeft: `calc(1em + ${theme.spacing(4)}px)`,
    transition: theme.transitions.create('width'),
    width: '100%',
    [theme.breakpoints.up('md')]: {
      width: '20ch',
    },
  },
  sectionDesktop: {
    display: 'flex',
  },
  modal: {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
    },
  paper: {
      backgroundColor: theme.palette.background.paper,
      border: '2px solid #000',
      boxShadow: theme.shadows[5],
      padding: theme.spacing(2, 4, 3),
  },
}));

export default function PrimarySearchAppBar(props : any) {
  const classes = useStyles();
  
  //dark mode 
  const {theme, setTheme} = props;
  const icon = !theme ? <Brightness7Icon /> : <Brightness3Icon />
  
  //modal
  const [open, setOpen] = useState<boolean>(false);

  const handleOpen = () => {
    setOpen(true);
  };
  const handleClose = () => {
    setOpen(false);
  };
  const contactModal = (
      <Modal
      aria-labelledby="transition-modal-title"
      aria-describedby="transition-modal-description"
      className={classes.modal}
      open={open}
      onClose={handleClose}
      closeAfterTransition
      BackdropComponent={Backdrop}
      BackdropProps={{
        timeout: 500,
      }}
    >
      <Fade in={open} >
        <div className={classes.paper}>
          <h1 id="transition-modal-title">Contact us</h1>
          <ContactusTable/>
        </div>
      </Fade>
    </Modal>
  );

  // github repository
  const repoLink : string = "https://github.com/NJYS/mask-cv-front"

  return (
    <div className={classes.grow}>
      <AppBar position="fixed">
        <Toolbar>
          <Typography className={classes.title} variant="h6" noWrap>
            TEAM NJYS - Mask CV App
          </Typography>
          <div className={classes.grow}/>
          <div className={classes.sectionDesktop}>
            <Tooltip title="day/night mode">
              <IconButton
                edge="end"
                color="inherit"
                aria-label="mode"
                onClick={() => setTheme(!theme)}
              >
                {icon}
              </IconButton>
            </Tooltip>
            <Box mr = "0.5rem"/>

            <Tooltip title="contact">
              <IconButton aria-label="contact" color="inherit" onClick={handleOpen}>
                <MailIcon />
              </IconButton>
            </Tooltip>
            {contactModal}

            <Tooltip title="github repo">
            <IconButton 
              aria-label="link to github"
              color="inherit" 
              onClick= {() => window.open(repoLink)}>
              <GitHubIcon />
            </IconButton>
            </Tooltip>
          </div>
        </Toolbar>
      </AppBar>
    </div>
  );
}
