import React, { useState, useMemo } from 'react';
import { BrowserRouter, Route, Link } from 'react-router-dom';

// pages
import Home from './Home';
import Realtime from './Realtime';

// components
import Appbar from './styles/Appbar';

// material UI
import { makeStyles, createMuiTheme, Theme, ThemeProvider } from '@material-ui/core/styles/';
import { Grid, CssBaseline, BottomNavigation, BottomNavigationAction, useMediaQuery } from '@material-ui/core/';
import ImageSearchIcon from '@material-ui/icons/ImageSearch';
import CenterFocusStrongIcon from '@material-ui/icons/CenterFocusStrong';

const useStyles = makeStyles((theme : Theme) => ({
  root : {
    width : 500,
  },
}));

function App() {

  const classes = useStyles();
  const prefersDarkMode : boolean = useMediaQuery('(prefers-color-scheme: dark)');
  const [theme, setTheme] = useState<boolean>(prefersDarkMode) // 다크모드
  const [value, setValue] = useState<number>(0);

  const appliedTheme = useMemo(
    () =>
      createMuiTheme({
        palette: {
          type: theme ? 'dark' : 'light',
        },
      }),
    [theme],
  );

  return (
    <>
    <BrowserRouter>
      <ThemeProvider theme={appliedTheme}>
        <CssBaseline/>
        <Appbar theme = {theme} setTheme = {setTheme}/>
        <Grid container spacing={0} direction="column" alignItems="center" justify="center">
          <Route exact path = "/" component = {Home}/>
          <Route path = "/realtime" component = {Realtime}/>
            <BottomNavigation value={value} 
              onChange={(e : React.ChangeEvent<{}>, newValue : number) => {setValue(newValue);}} 
              showLabels className={classes.root}>
              <BottomNavigationAction component = {Link} to ="/" label="Classify" icon={<ImageSearchIcon />} />
              <BottomNavigationAction component = {Link} to ="/realtime" label="Seg-Det" icon={<CenterFocusStrongIcon />} />
            </BottomNavigation>
        </Grid> 
      </ThemeProvider>
    </BrowserRouter>
    </>
  );
}

export default App;