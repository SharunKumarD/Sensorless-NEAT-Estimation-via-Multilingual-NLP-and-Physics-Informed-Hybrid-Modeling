import React, { useState, useEffect } from 'react';
import { View, Text, TextInput, TouchableOpacity, StyleSheet, Alert, SafeAreaView } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { API_BASE_URL } from '../config';

export default function ProfileScreen({ navigation }: any) {
  const [height, setHeight] = useState('');
  const [weight, setWeight] = useState('');
  const [age, setAge] = useState('');
  const [gender, setGender] = useState('');

  useEffect(() => {
    loadProfile();
  }, []);

  const loadProfile = async () => {
    try {
      const storedHeight = await AsyncStorage.getItem('user_height');
      const storedWeight = await AsyncStorage.getItem('user_weight');
      const storedAge = await AsyncStorage.getItem('user_age');
      const storedGender = await AsyncStorage.getItem('user_gender');
      
      if (storedHeight) setHeight(storedHeight);
      if (storedWeight) setWeight(storedWeight);
      if (storedAge) setAge(storedAge);
      if (storedGender) setGender(storedGender);
    } catch (error) {
      console.log('Error loading profile', error);
    }
  };

  const saveProfile = async () => {
    try {
      await AsyncStorage.setItem('user_height', height);
      await AsyncStorage.setItem('user_weight', weight);
      await AsyncStorage.setItem('user_age', age);
      await AsyncStorage.setItem('user_gender', gender);
      await AsyncStorage.setItem('user_id', 'local_user_1'); // Mock User ID
      
      const payload = {
        user_id: 'local_user_1',
        height: parseFloat(height) || 0,
        weight: parseFloat(weight) || 0,
        age: parseInt(age) || 0,
        gender: gender
      };
      
      await fetch(`${API_BASE_URL}/users`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      
      Alert.alert('Success', 'Profile saved securely!');
    } catch (error) {
      Alert.alert('Error', 'Failed to save profile');
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <Text style={styles.title}>Biomechanical Profile</Text>
      <Text style={styles.subtitle}>Calibrate your personal metabolic limits.</Text>
      
      <View style={styles.form}>
        <TextInput style={styles.input} placeholder="Weight (kg)" value={weight} onChangeText={setWeight} keyboardType="numeric" />
        <TextInput style={styles.input} placeholder="Height (cm)" value={height} onChangeText={setHeight} keyboardType="numeric" />
        <TextInput style={styles.input} placeholder="Age" value={age} onChangeText={setAge} keyboardType="numeric" />
        <TextInput style={styles.input} placeholder="Gender" value={gender} onChangeText={setGender} />
        
        <TouchableOpacity style={styles.saveButton} onPress={saveProfile}>
          <Text style={styles.saveButtonText}>Save Profile</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F5F9FF', padding: 24, justifyContent: 'center' },
  title: { fontSize: 28, fontWeight: 'bold', color: '#333', marginBottom: 8 },
  subtitle: { fontSize: 16, color: '#666', marginBottom: 24 },
  form: { backgroundColor: '#fff', borderRadius: 16, padding: 24, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.1, shadowRadius: 8, elevation: 5 },
  input: { backgroundColor: '#f0f0f0', borderRadius: 8, padding: 12, marginBottom: 16, fontSize: 16 },
  saveButton: { backgroundColor: '#007BFF', borderRadius: 12, padding: 16, alignItems: 'center', marginTop: 8 },
  saveButtonText: { color: '#fff', fontSize: 18, fontWeight: 'bold' }
});
