import { ref, push, set } from 'firebase/database';
import { database } from './firebase';

export const uploadFileToFirebase = async (file, fileCategory, fileType) => {
  try {
    const timestamp = Date.now();
    
    const reader = new FileReader();
    const base64Data = await new Promise((resolve, reject) => {
      reader.onloadend = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
    
    const metadata = {
      fileName: file.name,
      fileType: fileType,
      fileCategory: fileCategory,
      fileSize: file.size,
      uploadedAt: timestamp,
      fileData: base64Data, 
      status: 'pending'
    };
    
    console.log('Saving to Database...');
    const dbRef = ref(database, `uploads/${fileCategory}s`);
    const newUploadRef = push(dbRef);
    await set(newUploadRef, metadata);
    
    console.log('Upload successful!');
    return {
      success: true,
      uploadId: newUploadRef.key,
      metadata: metadata
    };
    
  } catch (error) {
    console.error('Error uploading file:', error);
    throw error;
  }
};

export const uploadAndWaitForResult = async (file, fileCategory, fileType, onProgress) => {
  try {
    const result = await uploadFileToFirebase(file, fileCategory, fileType);
    
    if (onProgress) {
      onProgress({ status: 'uploaded', data: result });
    }
    
    return result;
  } catch (error) {
    if (onProgress) {
      onProgress({ status: 'error', error: error.message });
    }
    throw error;
  }
};
