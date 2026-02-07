import { initializeApp } from 'firebase/app';
import { getDatabase } from 'firebase/database';

const firebaseConfig = {
  apiKey: "AIzaSyC5M_VW0avbD-1WIeKVNhuS_Gp_l9hZYqU",
  authDomain: "deep-busters.firebaseapp.com",
  databaseURL: "https://deep-busters-default-rtdb.asia-southeast1.firebasedatabase.app/",
  projectId: "deep-busters",
  storageBucket: "deep-busters.appspot.com",
  messagingSenderId: "453462781021",
  appId: "1:453462781021:web:1d9340319027dc236cf1b4"
};

const app = initializeApp(firebaseConfig);

export const database = getDatabase(app);

export default app;
