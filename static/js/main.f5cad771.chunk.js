(this["webpackJsonpmask-cv-front"]=this["webpackJsonpmask-cv-front"]||[]).push([[0],{100:function(e,t,n){"use strict";(function(e){var a=n(44),c=n.n(a),r=n(101),i=n(10),o=n(31),s=n(0),l=n(58),j=n.n(l),b=n(66),u=n(102),d=n.n(u),h=n(107),m=n(109),O=n(40),f=n(80),p=n(206),x=n(28),g=n(207),v=n(122),w=n(124),k=n(86),y=n(204),C=n(84),S=n(205),z=n(123),N=n(112),I=n.n(N),F=n(111),E=n.n(F),R=n(2),L=Object(y.a)((function(e){return{root:{width:500},button:{"& > *":{margin:e.spacing(1)}},input:{display:"none"},main:{padding:"2em",backgroundColor:"#757de8",height:"60vh",overflow:"auto"},progressCircle:{margin:"1em"}}})),M=Object(C.a)();M.typography.h3=Object(o.a)({fontSize:"1.2rem","@media (min-width:600px)":{fontSize:"1.5rem"}},M.breakpoints.up("md"),{fontSize:"2rem"}),t.a=function(){var t=L(),n=Object(s.useState)(""),a=Object(i.a)(n,2),o=a[0],l=a[1],u=Object(s.useState)(!1),y=Object(i.a)(u,2),C=y[0],N=y[1],F=Object(s.useState)(""),A=Object(i.a)(F,2),J=A[0],T=A[1],B=Object(s.useState)(!1),D=Object(i.a)(B,2),P=D[0],W=D[1],_=Object(s.useState)(),H=Object(i.a)(_,2),V=H[0],G=H[1],U="njys".concat(":","1q2w3e4r!"),Y={Authorization:"Basic "+e.from(U).toString("base64")},q=j.a.create({baseURL:"https://boostcamp-nyjs.herokuapp.com/",headers:Y}),$=Object(b.a)((function(e){return q.post("masks/",e)}),{onSuccess:function(e){T(e.data.result)},onError:function(){T("\uc804\uc1a1 \uc624\ub958")}}),K=Object(i.a)($,2),Q=K[0],X=K[1].isLoading;Object(s.useEffect)((function(){P?(l(""),N(!1)):""!==o&&N(!0)}),[P,o,l,N]);var Z=function(){var e=Object(r.a)(c.a.mark((function e(t){var n,a,r;return c.a.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:if(t.preventDefault(),n=t.target,G(n),P&&W(!1),e.prev=4,!n){e.next=13;break}return a=n.files[0],e.next=9,Object(h.a)(a);case 9:r=e.sent,T(""),l(r),N(!0);case 13:e.next=20;break;case 15:e.prev=15,e.t0=e.catch(4),T("\uc5c5\ub85c\ub4dc \uc624\ub958"),l(""),N(!1);case 20:case"end":return e.stop()}}),e,null,[[4,15]])})));return function(t){return e.apply(this,arguments)}}(),ee=function(){return Object(R.jsx)(O.a,{display:"flex",alignItems:"center",justifyContent:"center",className:t.progressCircle,children:Object(R.jsx)(f.a,{color:"primary"})})},te=function(){return Object(R.jsxs)(S.a,{theme:M,children:[Object(R.jsx)("br",{}),Object(R.jsx)(O.a,{display:"flex",alignItems:"center",justifyContent:"center",children:Object(R.jsx)(p.a,{variant:"h3",id:"res",children:J})})]})},ne=function(){T(""),W((function(e){return!e})),void 0!==V&&(V.value="")};return Object(R.jsxs)(R.Fragment,{children:[Object(R.jsxs)(x.a,{container:!0,spacing:0,direction:"column",alignItems:"center",justify:"center",children:[Object(R.jsx)(O.a,{mt:"10rem"}),Object(R.jsx)(g.a,{maxWidth:"sm",children:Object(R.jsx)(x.a,{item:!0,children:Object(R.jsxs)(v.a,{elevation:4,className:t.main,children:[P?Object(R.jsx)(m.a,{setPreview:l,camToggle:ne}):null,C?Object(R.jsx)(d.a,{src:o}):null,Object(R.jsx)(O.a,{mt:"3rem"}),X?Object(R.jsx)(ee,{}):Object(R.jsx)(te,{})]})})}),Object(R.jsx)(O.a,{mb:"0.5rem"}),Object(R.jsxs)(x.a,{container:!0,spacing:5,direction:"row",alignItems:"center",justify:"center",children:[Object(R.jsx)(x.a,{item:!0,children:Object(R.jsxs)("label",{htmlFor:"image_uploads",children:[Object(R.jsx)(w.a,{title:"image upload",placement:"left",arrow:!0,children:Object(R.jsx)(z.a,{color:"primary","aria-label":"upload",component:"span",children:Object(R.jsx)(E.a,{fontSize:"large"})})}),Object(R.jsx)("input",{type:"file",id:"image_uploads",accept:"image/*",onClick:function(){T(""),l("")},onChange:Z,className:t.input})]})}),Object(R.jsx)(x.a,{item:!0,children:Object(R.jsx)("label",{htmlFor:"open_webcam",children:Object(R.jsx)(w.a,{title:"webcam on",arrow:!0,children:Object(R.jsx)(z.a,{color:"primary","aria-label":"open_webcam",component:"span",onClick:ne,children:Object(R.jsx)(I.a,{fontSize:"large"})})})})}),Object(R.jsx)(x.a,{item:!0,children:Object(R.jsx)(k.a,{variant:"contained",size:"small",color:"primary",onClick:function(e){if(e.preventDefault(),""!==o){var t=JSON.stringify({image:o}),n=JSON.parse(t);Q(n)}else T("\ud30c\uc77c\uc774 \uc5c6\uc2b5\ub2c8\ub2e4.")},children:"\u2728\uc81c\ucd9c"})})]})]}),Object(R.jsx)(O.a,{mb:"0.5rem"})]})}}).call(this,n(160).Buffer)},107:function(e,t,n){"use strict";var a=n(108),c=n.n(a);t.a=function(e){return new Promise((function(t){c.a.imageFileResizer(e,448,448,"JPEG",100,0,(function(e){t(e.toString())}),"base64")}))}},109:function(e,t,n){"use strict";var a=n(0),c=n(64),r=n.n(c),i=n(28),o=n(40),s=n(86),l=n(2),j={width:448,height:448,facingMode:"user"};t.a=function(e){var t=Object(a.useRef)(null),n=e.camToggle,c=e.setPreview,b=Object(a.useCallback)((function(){var e=t.current.getScreenshot();c(e),n()}),[t,n,c]);return Object(l.jsxs)(l.Fragment,{children:[Object(l.jsx)(i.a,{container:!0,justify:"center",alignItems:"center",children:Object(l.jsx)(r.a,{audio:!1,height:window.innerHeight>800?448:224,ref:t,screenshotFormat:"image/jpeg",width:window.innerWidth>800?448:224,videoConstraints:j})}),Object(l.jsx)(o.a,{mt:"1rem"}),Object(l.jsxs)(i.a,{container:!0,spacing:5,direction:"row",alignItems:"center",justify:"center",children:[Object(l.jsx)(i.a,{item:!0,children:Object(l.jsx)(s.a,{variant:"contained",size:"small",color:"primary",onClick:b,children:"\ucea1\uccd0"})}),Object(l.jsx)(i.a,{item:!0,children:Object(l.jsx)(s.a,{variant:"contained",size:"small",color:"primary",onClick:n,children:"\uc885\ub8cc"})})]})]})}},200:function(e,t,n){"use strict";n.r(t);var a=n(0),c=n.n(a),r=n(14),i=n.n(r),o=n(10),s=n(73),l=n(13),j=n(100),b=n(58),u=n.n(b),d=n(66),h=n(64),m=n.n(h),O=function(e,t){var n=Object(a.useRef)();Object(a.useEffect)((function(){n.current=e}),[e]),Object(a.useEffect)((function(){if(null!==t){var e=setInterval((function(){n.current()}),t);return function(){return clearInterval(e)}}}),[t])},f=n(28),p=n(40),x=n(69),g=n(68),v=n(232),w=n(233),k=n(67),y=n(71),C=n(234),S=n(235),z=n(236),N=n(72),I=n(237),F=n(238),E=n(239),R=n(240),L=n(70),M=n(241),A=n(39),J=n(2),T=[x.a,g.a,v.a,w.a,k.a,y.a,C.a,S.a,z.a,N.a,I.a,F.a,E.a,R.a,L.a,M.a],B={check:!0,bboxes:[[0,0,0,0]],labels:[""],segmentations:["0,0"]},D=[x.a[400],v.a[400],k.a[400],y.a[400],S.a[400],N.a[400],F.a[400],E.a[400],R.a[400],M.a[400]],P={width:448,height:448,facingMode:"user"},W=function(e){var t=Object(a.useRef)(null),n=Object(a.useRef)(null),c=Object(a.useState)(""),r=Object(o.a)(c,2),i=r[0],s=r[1],l=Object(a.useState)(D),j=Object(o.a)(l,2),b=j[0],h=j[1],x=Object(a.useState)(B),g=Object(o.a)(x,2),v=g[0],w=g[1],k=Object(a.useState)(B.labels),y=Object(o.a)(k,2),C=y[0],S=y[1],z=Object(a.useState)(B.bboxes),N=Object(o.a)(z,2),I=N[0],F=N[1],E=Object(a.useState)(B.segmentations),R=Object(o.a)(E,2),L=R[0],M=R[1],W=u.a.create({baseURL:"http://54.180.91.142/"}),_=Object(d.a)((function(e){return W.post("masks",e)}),{onSuccess:function(e){var t=JSON.parse(e.data);t.check?w(t):V()},onError:function(e){console.log(e)}}),H=Object(o.a)(_,1)[0],V=Object(a.useCallback)((function(){for(var e=[],t=0;t<10;t++)e.push(T[Math.floor(14*Math.random())][100*(Math.floor(8*Math.random())+1)]);h(e)}),[h]),G=Object(a.useCallback)((function(){var e=n.current.getScreenshot();s(e)}),[n,s]);return Object(a.useEffect)((function(){S(v.labels),F(v.bboxes),M(v.segmentations)}),[v,S,F,M]),Object(a.useEffect)((function(){t.current&&e.classification?Object(A.a)(t.current).selectAll("text").data(I).join((function(e){return e.append("text")}),(function(e){return e.attr("class","updated")}),(function(e){return e.remove()})).attr("x",(function(e){return e[0]-15})).attr("y",(function(e){return e[1]-5})).data(C).text((function(e){return e})).attr("font-family","Arial").attr("font-size","11px").attr("text-align","center").attr("fill","red").attr("stroke-width",1):Object(A.a)("text").remove()}),[I,C,e.classification]),Object(a.useEffect)((function(){t.current&&e.detection?Object(A.a)(t.current).selectAll("rect").data(I).join((function(e){return e.append("rect")}),(function(e){return e.attr("class","updated")}),(function(e){return e.remove()})).attr("width",(function(e){return e[2]-e[0]})).attr("height",(function(e){return e[3]-e[1]})).attr("x",(function(e){return e[0]})).attr("y",(function(e){return e[1]})).attr("fill","transparent").attr("stroke","red").attr("stroke-width",2):Object(A.a)("rect").remove()}),[I,e.detection]),Object(a.useEffect)((function(){if(t.current&&e.segmentation){Object(A.a)(t.current).selectAll("polygon").data(L).join((function(e){return e.append("polygon")}),(function(e){return e.attr("class","updated")}),(function(e){return e.remove()})).attr("points",(function(e){return e})).data([0,1,2,3,4,5,6,7,8,9]).attr("fill",(function(e){return b[e]})).attr("stroke-width",2).attr("opacity",.5)}else Object(A.a)("polygon").remove()}),[L,b,e.segmentation]),O((function(){if(n.current)try{G(),function(){var e=JSON.stringify({data:i}),t=JSON.parse(e);""!==i&&H(t)}()}catch(e){console.log(e)}}),1e3/(isNaN(Number(e.interval))?1:Number(e.interval))),Object(a.useEffect)((function(){return function(){return s("")}}),[]),Object(J.jsxs)(J.Fragment,{children:[Object(J.jsxs)(f.a,{container:!0,justify:"center",alignItems:"center",children:[Object(J.jsx)(m.a,{audio:!1,height:window.innerHeight>640?448:224,ref:n,screenshotFormat:"image/jpeg",width:window.innerWidth>640?448:224,videoConstraints:P}),Object(J.jsx)("svg",{ref:t,style:{position:"absolute",marginLeft:"auto",marginRight:"auto",textAlign:"center",zIndex:9,width:window.innerWidth>640?448:224,height:window.innerHeight>640?448:224}})]}),Object(J.jsx)(p.a,{mt:"1rem"})]})},_=n(86),H=n(207),V=n(123),G=n(5),U=n(204),Y=n(124),q=n(122),$=n(114),K=n.n($),Q=n(113),X=n.n(Q),Z=n(257),ee=n(206),te=Object(G.a)({root:{color:"#52af77",height:8},thumb:{height:24,width:24,backgroundColor:"#fff",border:"2px solid currentColor",marginTop:-8,marginLeft:-12,"&:focus, &:hover, &$active":{boxShadow:"inherit"}},active:{},valueLabel:{left:"calc(-50% + 4px)"},track:{height:8,borderRadius:4},rail:{height:8,borderRadius:4}})(Z.a),ne=Object(U.a)((function(e){return{root:{width:500},button:{"& > *":{margin:e.spacing(1)}},input:{display:"none"},main:{padding:"2em",backgroundColor:"#757de8",dark:"#002984",height:"60vh",overflow:"auto"},progressCircle:{margin:"1em"}}}));var ae=function(){var e=ne(),t=Object(a.useState)(!1),n=Object(o.a)(t,2),c=n[0],r=n[1],i=Object(a.useState)(!1),s=Object(o.a)(i,2),l=s[0],j=s[1],b=Object(a.useState)(!1),u=Object(o.a)(b,2),d=u[0],h=u[1],m=Object(a.useState)(!1),O=Object(o.a)(m,2),x=O[0],g=O[1],v=Object(a.useState)(7),w=Object(o.a)(v,2),k=w[0],y=w[1],C=function(){j((function(e){return!e}))},S=function(){h((function(e){return!e}))},z=function(){g((function(e){return!e}))};return Object(J.jsxs)(J.Fragment,{children:[Object(J.jsxs)(f.a,{container:!0,spacing:0,direction:"column",alignItems:"center",justify:"center",children:[Object(J.jsx)(p.a,{mt:"10rem"}),Object(J.jsx)(H.a,{maxWidth:"sm",children:Object(J.jsx)(f.a,{item:!0,children:Object(J.jsxs)(q.a,{elevation:4,className:e.main,children:[c?Object(J.jsx)(W,{interval:k,classification:l,segmentation:d,detection:x}):null,Object(J.jsx)(p.a,{mt:"3rem"})]})})}),Object(J.jsx)(p.a,{mb:"0.5rem"}),Object(J.jsxs)(f.a,{container:!0,spacing:3,direction:"row",alignItems:"center",justify:"center",children:[Object(J.jsx)(f.a,{item:!0,children:Object(J.jsx)(Y.a,{title:"classification",arrow:!0,children:l?Object(J.jsx)(_.a,{variant:"contained",size:"small",color:"secondary",onClick:C,children:"class\ud83e\udde1"}):Object(J.jsx)(_.a,{variant:"contained",size:"small",color:"primary",onClick:C,children:"class\ud83e\udde1"})})}),Object(J.jsx)(f.a,{item:!0,children:Object(J.jsx)(Y.a,{title:"segmentation",arrow:!0,children:d?Object(J.jsx)(_.a,{variant:"contained",size:"small",color:"secondary",onClick:S,children:"seg\ud83d\udc9b"}):Object(J.jsx)(_.a,{variant:"contained",size:"small",color:"primary",onClick:S,children:"seg\ud83d\udc9b"})})}),Object(J.jsx)(f.a,{item:!0,children:Object(J.jsx)(Y.a,{title:"detection",arrow:!0,children:x?Object(J.jsx)(_.a,{variant:"contained",size:"small",color:"secondary",onClick:z,children:"det\ud83d\udc99"}):Object(J.jsx)(_.a,{variant:"contained",size:"small",color:"primary",onClick:z,children:"det\ud83d\udc99"})})}),Object(J.jsx)(f.a,{item:!0,children:Object(J.jsx)("label",{htmlFor:"open_webcam",children:Object(J.jsx)(Y.a,{title:"webcam on/off",arrow:!0,children:Object(J.jsx)(V.a,{color:"primary","aria-label":"open_webcam",component:"span",onClick:function(){r((function(e){return!e}))},children:c?Object(J.jsx)(X.a,{fontSize:"large"}):Object(J.jsx)(K.a,{fontSize:"large"})})})})}),Object(J.jsxs)(f.a,{item:!0,children:[Object(J.jsx)(ee.a,{gutterBottom:!0,children:"\u3000\u3000fps \uc124\uc815\u3000\u3000"}),Object(J.jsx)(te,{valueLabelDisplay:"auto","aria-label":"pretto slider",defaultValue:7,min:1,max:15,step:1,onChange:function(e,t){y(t)}})]})]})]}),Object(J.jsx)(p.a,{mb:"0.5rem"})]})},ce=n(31),re=n(256),ie=n(242),oe=n(243),se=n(244),le=n(245),je=n(246),be=n(247),ue=Object(G.a)((function(e){return Object(re.a)({head:{backgroundColor:e.palette.common.black,color:e.palette.common.white},body:{fontSize:14}})}))(ie.a),de=Object(G.a)((function(e){return Object(re.a)({root:{"&:nth-of-type(odd)":{backgroundColor:e.palette.action.hover}}})}))(oe.a);function he(e,t,n,a){return{name:e,part:t,mail:n,github:a}}var me=[he("\uae40\uaddc\ube48","model,data","kimkyu1515@naver.com","https://github.com/kkbwilldo"),he("\uad8c\ud0dc\ud655","model,data","taehwak@hanyang.ac.kr","https://github.com/taehwakkwon"),he("\uae40\uc0c1\ud6c8","front","ropeiny@gmail.com","https://github.com/simon-hoon"),he("\ubc15\uacbd\ud658","back","john1725258@gmail.com","https://github.com/hwan1753"),he("\uc804\uc8fc\uc601","front","zhonya_j@g.seoultech.ac.kr","https://github.com/zhonya-j")],Oe=Object(U.a)({table:{minWidth:700}});function fe(){var e=Oe();return Object(J.jsx)(se.a,{component:q.a,children:Object(J.jsxs)(le.a,{className:e.table,"aria-label":"customized table",children:[Object(J.jsx)(je.a,{children:Object(J.jsxs)(oe.a,{children:[Object(J.jsx)(ue,{children:"name"}),Object(J.jsx)(ue,{align:"center",children:"part\xa0"}),Object(J.jsx)(ue,{align:"center",children:"mail\xa0"}),Object(J.jsx)(ue,{align:"center",children:"github\xa0"})]})}),Object(J.jsx)(be.a,{children:me.map((function(e){return Object(J.jsxs)(de,{children:[Object(J.jsx)(ue,{component:"th",scope:"row",children:e.name}),Object(J.jsx)(ue,{align:"center",children:e.part}),Object(J.jsx)(ue,{align:"center",children:e.mail}),Object(J.jsx)(ue,{align:"center",children:e.github})]},e.name)}))})]})})}var pe=n(255),xe=n(248),ge=n(208),ve=n(249),we=n(250),ke=n(116),ye=n.n(ke),Ce=n(115),Se=n.n(Ce),ze=n(117),Ne=n.n(ze),Ie=n(118),Fe=n.n(Ie),Ee=Object(U.a)((function(e){return{grow:{flexGrow:1},menuButton:{marginRight:e.spacing(2)},title:{display:"block"},inputRoot:{color:"inherit"},inputInput:Object(ce.a)({padding:e.spacing(1,1,1,0),paddingLeft:"calc(1em + ".concat(e.spacing(4),"px)"),transition:e.transitions.create("width"),width:"100%"},e.breakpoints.up("md"),{width:"20ch"}),sectionDesktop:{display:"flex"},modal:{display:"flex",alignItems:"center",justifyContent:"center"},paper:{backgroundColor:e.palette.background.paper,border:"2px solid #000",boxShadow:e.shadows[5],padding:e.spacing(2,4,3)}}}));function Re(e){var t=Ee(),n=e.theme,c=e.setTheme,r=n?Object(J.jsx)(ye.a,{}):Object(J.jsx)(Se.a,{}),i=Object(a.useState)(!1),s=Object(o.a)(i,2),l=s[0],j=s[1],b=Object(J.jsx)(pe.a,{"aria-labelledby":"transition-modal-title","aria-describedby":"transition-modal-description",className:t.modal,open:l,onClose:function(){j(!1)},closeAfterTransition:!0,BackdropComponent:xe.a,BackdropProps:{timeout:500},children:Object(J.jsx)(ge.a,{in:l,children:Object(J.jsxs)("div",{className:t.paper,children:[Object(J.jsx)("h1",{id:"transition-modal-title",children:"Contact us"}),Object(J.jsx)(fe,{})]})})});return Object(J.jsx)("div",{className:t.grow,children:Object(J.jsx)(ve.a,{position:"fixed",children:Object(J.jsxs)(we.a,{children:[Object(J.jsx)(ee.a,{className:t.title,variant:"h6",noWrap:!0,children:"TEAM NJYS - Mask CV App"}),Object(J.jsx)("div",{className:t.grow}),Object(J.jsxs)("div",{className:t.sectionDesktop,children:[Object(J.jsx)(Y.a,{title:"day/night mode",children:Object(J.jsx)(V.a,{edge:"end",color:"inherit","aria-label":"mode",onClick:function(){return c(!n)},children:r})}),Object(J.jsx)(p.a,{mr:"0.5rem"}),Object(J.jsx)(Y.a,{title:"contact",children:Object(J.jsx)(V.a,{"aria-label":"contact",color:"inherit",onClick:function(){j(!0)},children:Object(J.jsx)(Ne.a,{})})}),b,Object(J.jsx)(Y.a,{title:"github repo",children:Object(J.jsx)(V.a,{"aria-label":"link to github",color:"inherit",onClick:function(){return window.open("https://github.com/NJYS/Mask-CV-App")},children:Object(J.jsx)(Fe.a,{})})})]})]})})})}var Le=n(84),Me=n(205),Ae=n(251),Je=n(252),Te=n(253),Be=n(254),De=n(119),Pe=n.n(De),We=n(120),_e=n.n(We),He=Object(U.a)((function(e){return{root:{width:500}}}));var Ve=function(){var e=He(),t=Object(Ae.a)("(prefers-color-scheme: dark)"),n=Object(a.useState)(t),c=Object(o.a)(n,2),r=c[0],i=c[1],b=Object(a.useState)(0),u=Object(o.a)(b,2),d=u[0],h=u[1],m=Object(a.useMemo)((function(){return Object(Le.a)({palette:{type:r?"dark":"light"}})}),[r]);return Object(J.jsx)(J.Fragment,{children:Object(J.jsx)(s.a,{children:Object(J.jsxs)(Me.a,{theme:m,children:[Object(J.jsx)(Je.a,{}),Object(J.jsx)(Re,{theme:r,setTheme:i}),Object(J.jsxs)(f.a,{container:!0,spacing:0,direction:"column",alignItems:"center",justify:"center",children:[Object(J.jsx)(l.a,{exact:!0,path:"/",component:j.a}),Object(J.jsx)(l.a,{path:"/realtime",component:ae}),Object(J.jsxs)(Te.a,{value:d,onChange:function(e,t){h(t)},showLabels:!0,className:e.root,children:[Object(J.jsx)(Be.a,{component:s.b,to:"/",label:"Classify",icon:Object(J.jsx)(Pe.a,{})}),Object(J.jsx)(Be.a,{component:s.b,to:"/realtime",label:"Seg-Det",icon:Object(J.jsx)(_e.a,{})})]})]})]})})})},Ge=function(e){e&&e instanceof Function&&n.e(3).then(n.bind(null,259)).then((function(t){var n=t.getCLS,a=t.getFID,c=t.getFCP,r=t.getLCP,i=t.getTTFB;n(e),a(e),c(e),r(e),i(e)}))};i.a.render(Object(J.jsx)(c.a.StrictMode,{children:Object(J.jsx)(Ve,{})}),document.getElementById("root")),Ge()}},[[200,1,2]]]);
//# sourceMappingURL=main.f5cad771.chunk.js.map