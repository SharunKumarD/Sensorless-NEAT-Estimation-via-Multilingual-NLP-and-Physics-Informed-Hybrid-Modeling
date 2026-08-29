import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';

// Import Screens
import DataInputScreen from './screens/DataInputScreen';
import DashboardScreen from './screens/DashboardScreen';

// Define the TypeScript types for the navigation stack
export type RootStackParamList = {
  DataInput: undefined;
  Dashboard: { dashboardData: any };
};

const Stack = createNativeStackNavigator<RootStackParamList>();

export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator 
        initialRouteName="DataInput"
        screenOptions={{
          headerStyle: {
            backgroundColor: '#F5F9FF', // Matching the app's clean white/light blue aesthetic
          },
          headerShadowVisible: false, // Removes the bottom border for a cleaner look
          headerTintColor: '#333',
          headerTitleStyle: {
            fontWeight: 'bold',
          },
        }}
      >
        <Stack.Screen 
          name="DataInput" 
          component={DataInputScreen} 
          options={{ title: 'NEAT Tracker' }}
        />
        <Stack.Screen 
          name="Dashboard" 
          component={DashboardScreen} 
          options={{ title: 'Analytics Dashboard' }}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
