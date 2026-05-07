import { JSDOM } from 'jsdom';
import React, { useEffect } from 'react';
import { createRoot } from 'react-dom/client';

const dom = new JSDOM('<!doctype html><html><body><div id="root"></div></body></html>', { url: 'http://localhost/' });
global.window = dom.window;
global.document = dom.window.document;
global.navigator = dom.window.navigator;

function Child(){
  useEffect(() => { console.log('child effect'); }, []);
  return React.createElement('div', null, 'child');
}
function Parent(){
  useEffect(() => { console.log('parent effect'); }, []);
  return React.createElement(Child);
}

const root = createRoot(document.getElementById('root'));
root.render(React.createElement(Parent));
setTimeout(() => process.exit(0), 50);
