// material-UI
import { withStyles, Theme, createStyles, makeStyles } from '@material-ui/core/styles';
import { Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper  } from '@material-ui/core/';

const StyledTableCell = withStyles((theme: Theme) =>
  createStyles({
    head: {
      backgroundColor: theme.palette.common.black,
      color: theme.palette.common.white,
    },
    body: {
      fontSize: 14,
    },
  }),
)(TableCell);

const StyledTableRow = withStyles((theme: Theme) =>
  createStyles({
    root: {
      '&:nth-of-type(odd)': {
        backgroundColor: theme.palette.action.hover,
      },
    },
  }),
)(TableRow);

function createData(name: string, part: string, mail: string, github: string) {
  return { name, part, mail, github};
}

const rows = [
  createData('김규빈', "model,data","kimkyu1515@naver.com","https://github.com/kkbwilldo"),
  createData('권태확', "model,data","taehwak@hanyang.ac.kr","https://github.com/taehwakkwon"),
  createData('김상훈', "front","ropeiny@gmail.com","https://github.com/simon-hoon"),
  createData('박경환', "back","john1725258@gmail.com","https://github.com/hwan1753"),
  createData('전주영', "front","zhonya_j@g.seoultech.ac.kr","https://github.com/zhonya-j"),
];

const useStyles = makeStyles({
  table: {
    minWidth: 700,
  },
});

export default function CustomizedTables() {
  const classes = useStyles();

  return (
    <TableContainer component={Paper}>
      <Table className={classes.table} aria-label="customized table">
        <TableHead>
          <TableRow>
            <StyledTableCell>name</StyledTableCell>
            <StyledTableCell align="center">part&nbsp;</StyledTableCell>
            <StyledTableCell align="center">mail&nbsp;</StyledTableCell>
            <StyledTableCell align="center">github&nbsp;</StyledTableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {rows.map((row) => (
            <StyledTableRow key={row.name}>
              <StyledTableCell component="th" scope="row">
                {row.name}
              </StyledTableCell>
              <StyledTableCell align="center">{row.part}</StyledTableCell>
              <StyledTableCell align="center">{row.mail}</StyledTableCell>
              <StyledTableCell align="center">{row.github}</StyledTableCell>
            </StyledTableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
}